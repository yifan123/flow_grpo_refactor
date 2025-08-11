import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from .utils import calculate_zero_std_ratio, unwrap_model


class GRPOMixin:
    """GRPO (Generalized Reward Preference Optimization) algorithm mixin."""
    
    def sample_batch(self, reward_fn):
        """Sample batch for GRPO training."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer.eval()
        
        info = defaultdict(list)
        
        for _ in range(self.config.sample.num_batches_per_epoch):
            batch = next(iter(self.dataloader))
            
            if hasattr(self, 'get_dataset_class_name') and self.get_dataset_class_name() == "GenevalPromptImageDataset":
                prompts, prompt_metadatas, images, prompt_with_image_paths = batch
            else:
                prompts, prompt_metadatas = batch
                images = None
                
            # Compute text embeddings
            if images is not None:
                embeddings = self.compute_text_embeddings(prompts, images)
            else:
                embeddings = self.compute_text_embeddings(prompts)
                
            # Generate samples
            with torch.no_grad():
                sample_info = self._generate_samples_grpo(prompts, embeddings, reward_fn)
                
            for key, value in sample_info.items():
                info[key].extend(value)
                
        # Convert to tensors
        for key in ["timesteps", "log_probs", "latents", "noise", "rewards"]:
            info[key] = torch.cat(info[key])
            
        transformer.train()
        return dict(info)
        
    def _generate_samples_grpo(self, prompts, embeddings, reward_fn):
        """Generate samples for GRPO."""
        info = defaultdict(list)
        
        batch_size = len(prompts)
        
        # Sample timesteps
        if self.config.sample.sample_timesteps_type == "uniform":
            timesteps = torch.randint(
                0, self.pipeline.scheduler.config.num_train_timesteps,
                (batch_size * self.config.sample.num_images_per_prompt,),
                device=self.accelerator.device
            )
        else:
            # Add other sampling strategies here
            timesteps = torch.randint(
                0, self.pipeline.scheduler.config.num_train_timesteps,
                (batch_size * self.config.sample.num_images_per_prompt,),
                device=self.accelerator.device
            )
            
        # Sample noise and latents
        latents = torch.randn(
            (batch_size * self.config.sample.num_images_per_prompt, 16, 128, 128),
            device=self.accelerator.device,
            dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        )
        
        noise = torch.randn_like(latents)
        
        # Prepare embeddings for batch
        if len(embeddings) == 4:  # Kontext with image_ids
            prompt_embeds, pooled_prompt_embeds, text_ids, image_ids = embeddings
            prompt_embeds = prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            text_ids = text_ids.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            image_ids = image_ids.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
        elif len(embeddings) == 3:  # FLUX with text_ids
            prompt_embeds, pooled_prompt_embeds, text_ids = embeddings
            prompt_embeds = prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            text_ids = text_ids.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
        else:  # SD3
            prompt_embeds, pooled_prompt_embeds = embeddings
            prompt_embeds = prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
            
        # Sample using SDE
        sde_kwargs = {
            "latents": latents,
            "noise": noise,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "timesteps": timesteps,
            "noise_scheduler": self.pipeline.scheduler,
            "noise_level": self.config.sample.noise_level
        }
        
        if len(embeddings) == 4:
            sde_kwargs["image_ids"] = image_ids
            sde_kwargs["text_ids"] = text_ids
        elif len(embeddings) == 3:
            sde_kwargs["text_ids"] = text_ids
            
        latents, log_probs = self.sample_sde_step(**sde_kwargs)
        
        # Decode images
        with torch.no_grad():
            images = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = (images * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            
        # Compute rewards
        rewards = reward_fn(images, [prompts[i // self.config.sample.num_images_per_prompt] for i in range(len(images))])
        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)
        
        # Collect info
        info["timesteps"].append(timesteps)
        info["log_probs"].append(log_probs)
        info["latents"].append(latents)
        info["noise"].append(noise)
        info["rewards"].append(rewards)
        
        return info
        
    def training_step(self, info):
        """GRPO training step."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        
        # Group rewards and compute advantages
        rewards = info["rewards"].cpu().numpy()
        group_rewards = rewards.reshape(-1, self.config.sample.num_images_per_prompt)
        group_rewards_mean = group_rewards.mean(axis=1, keepdims=True)
        advantages = (group_rewards - group_rewards_mean).flatten()
        advantages = torch.tensor(advantages, device=self.accelerator.device, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Compute ratio and loss
        with self.accelerator.accumulate(transformer):
            # Get current log probs
            kwargs = {
                "latents": info["latents"],
                "noise": info["noise"], 
                "timesteps": info["timesteps"],
                "noise_scheduler": self.pipeline.scheduler,
                "with_grad": True
            }
            
            # Add embedding info based on model type
            batch_prompts = [f"prompt_{i}" for i in range(len(info["latents"]))]  # Placeholder
            embeddings = self.compute_text_embeddings(batch_prompts[:len(info["latents"]) // self.config.sample.num_images_per_prompt])
            
            if len(embeddings) == 4:  # Kontext
                kwargs.update({
                    "prompt_embeds": embeddings[0].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "pooled_prompt_embeds": embeddings[1].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "text_ids": embeddings[2].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "image_ids": embeddings[3].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
                })
            elif len(embeddings) == 3:  # FLUX
                kwargs.update({
                    "prompt_embeds": embeddings[0].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "pooled_prompt_embeds": embeddings[1].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "text_ids": embeddings[2].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
                })
            else:  # SD3
                kwargs.update({
                    "prompt_embeds": embeddings[0].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0),
                    "pooled_prompt_embeds": embeddings[1].repeat_interleave(self.config.sample.num_images_per_prompt, dim=0)
                })
                
            current_log_probs = self.compute_log_prob(**kwargs)
            
            # Compute policy gradient loss
            ratio = torch.exp(current_log_probs - info["log_probs"].detach())
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.train.cliprange, 1.0 + self.config.train.cliprange)
            
            loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Add KL penalty if configured
            if self.config.train.beta > 0:
                kl_penalty = self.config.train.beta * (current_log_probs - info["log_probs"].detach()).pow(2).mean()
                loss += kl_penalty
                
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(transformer.parameters(), self.config.train.max_grad_norm)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean(),
            "std_reward": rewards.std(),
            "mean_ratio": ratio.mean().item(),
        }
        
    def compute_loss(self, *args, **kwargs):
        """GRPO loss computation - handled in training_step."""
        pass


class DPOMixin:
    """DPO (Direct Preference Optimization) algorithm mixin."""
    
    def setup_lora(self):
        """Setup dual LoRA adapters for DPO."""
        super().setup_lora()
        
        # Create reference adapter
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer.add_adapter("reference", transformer.peft_config["default"])
        transformer.set_adapter("default")
        
    def sample_batch(self, reward_fn):
        """Sample batch for DPO training."""
        # Implementation would be similar to GRPO but with preference pairs
        pass
        
    def compute_loss(self, chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, rejected_ref_log_probs):
        """Compute DPO loss."""
        chosen_relative_log_probs = chosen_log_probs - chosen_ref_log_probs
        rejected_relative_log_probs = rejected_log_probs - rejected_ref_log_probs
        
        loss = -F.logsigmoid(self.config.train.beta * (chosen_relative_log_probs - rejected_relative_log_probs)).mean()
        return loss
        
    def training_step(self, info):
        """DPO training step."""
        # Implementation would handle preference pairs
        pass


class SFTMixin:
    """SFT (Supervised Fine-Tuning) algorithm mixin."""
    
    def sample_batch(self, reward_fn):
        """Sample batch for SFT training."""
        # SFT doesn't need reward sampling
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer.eval()
        
        batch = next(iter(self.dataloader))
        prompts, prompt_metadatas = batch
        
        # Compute text embeddings
        embeddings = self.compute_text_embeddings(prompts)
        
        # Sample noise and timesteps for SFT
        batch_size = len(prompts)
        latents = torch.randn(
            (batch_size, 16, 128, 128),
            device=self.accelerator.device,
            dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        )
        
        timesteps = torch.randint(
            0, self.pipeline.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device
        )
        
        noise = torch.randn_like(latents)
        
        transformer.train()
        
        return {
            "prompts": prompts,
            "embeddings": embeddings,
            "latents": latents,
            "timesteps": timesteps,
            "noise": noise
        }
        
    def compute_loss(self, model_pred, target):
        """Compute MSE loss for SFT."""
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
    def training_step(self, info):
        """SFT training step."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        
        with self.accelerator.accumulate(transformer):
            # Add noise to latents
            noisy_latents = self.pipeline.scheduler.add_noise(
                info["latents"], info["noise"], info["timesteps"]
            )
            
            # Get target
            if hasattr(self.pipeline.scheduler.config, 'prediction_type'):
                if self.pipeline.scheduler.config.prediction_type == "epsilon":
                    target = info["noise"]
                elif self.pipeline.scheduler.config.prediction_type == "v_prediction":
                    target = self.pipeline.scheduler.get_velocity(info["latents"], info["noise"], info["timesteps"])
                else:
                    raise ValueError(f"Unknown prediction type {self.pipeline.scheduler.config.prediction_type}")
            else:
                target = info["noise"]  # Default to epsilon prediction
                
            # Forward pass
            embeddings = info["embeddings"]
            if len(embeddings) == 2:  # SD3
                model_pred = transformer(
                    noisy_latents,
                    info["timesteps"],
                    encoder_hidden_states=embeddings[0],
                    pooled_projections=embeddings[1]
                ).sample
            else:  # FLUX variants
                model_pred = transformer(
                    noisy_latents,
                    info["timesteps"], 
                    encoder_hidden_states=embeddings[0],
                    pooled_projections=embeddings[1],
                    # Additional args for FLUX
                ).sample
                
            loss = self.compute_loss(model_pred, target)
            
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(transformer.parameters(), self.config.train.max_grad_norm)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return {"loss": loss.item()}