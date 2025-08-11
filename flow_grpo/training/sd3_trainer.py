import torch
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from .base_trainer import BaseTrainer


class SD3Trainer(BaseTrainer):
    """Stable Diffusion 3 trainer implementation."""
    
    def load_pipeline(self):
        """Load SD3 pipeline."""
        return StableDiffusion3Pipeline.from_pretrained(
            self.config.pretrained.model, torch_dtype=torch.float16
        )
        
    def load_model_components(self):
        """Load SD3 model components."""
        # Components are already loaded with the pipeline
        pass
        
    def get_lora_target_modules(self):
        """Get LoRA target modules for SD3."""
        return ["to_k", "to_q", "to_v", "to_out.0"]
        
    def get_transformer_attr_name(self):
        """Get transformer attribute name for SD3."""
        return "transformer"
        
    def setup_text_encoders(self):
        """Setup SD3 text encoders."""
        self.pipeline.text_encoder.to(self.accelerator.device, dtype=torch.float16)
        self.pipeline.text_encoder_2.to(self.accelerator.device, dtype=torch.float16)
        self.pipeline.text_encoder_3.to(self.accelerator.device, dtype=torch.float16)
        
    def compute_text_embeddings(self, prompts):
        """Compute SD3 text embeddings."""
        return encode_prompt(
            self.pipeline,
            prompts,
            self.accelerator.device,
            do_classifier_free_guidance=False,
        )
        
    def compute_log_prob(self, latents, noise, prompt_embeds, pooled_prompt_embeds, 
                        timesteps, noise_scheduler, with_grad=False):
        """Compute log probabilities for SD3."""
        return pipeline_with_logprob(
            self.pipeline,
            latents=latents, 
            noise=noise,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            with_grad=with_grad
        )
        
    def sample_sde_step(self, latents, noise, prompt_embeds, pooled_prompt_embeds, 
                       timesteps, noise_scheduler, noise_level=0.1):
        """Sample using SDE step for SD3."""
        return sde_step_with_logprob(
            self.pipeline,
            latents=latents,
            noise=noise, 
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            noise_level=noise_level
        )