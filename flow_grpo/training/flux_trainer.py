import torch
from diffusers import FluxPipeline
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
from .base_trainer import BaseTrainer


class FluxTrainer(BaseTrainer):
    """FLUX model trainer implementation."""
    
    def load_pipeline(self):
        """Load FLUX pipeline."""
        return FluxPipeline.from_pretrained(
            self.config.pretrained.model, torch_dtype=torch.bfloat16
        )
        
    def load_model_components(self):
        """Load FLUX model components."""
        # Components are already loaded with the pipeline
        pass
        
    def get_lora_target_modules(self):
        """Get LoRA target modules for FLUX."""
        return ["to_k", "to_q", "to_v", "to_out.0", "proj_mlp"]
        
    def get_transformer_attr_name(self):
        """Get transformer attribute name for FLUX."""
        return "transformer"
        
    def setup_text_encoders(self):
        """Setup FLUX text encoders."""
        self.pipeline.text_encoder.to(self.accelerator.device, dtype=torch.bfloat16)
        self.pipeline.text_encoder_2.to(self.accelerator.device, dtype=torch.bfloat16)
        
    def compute_text_embeddings(self, prompts):
        """Compute FLUX text embeddings."""
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            self.pipeline,
            prompts,
            self.accelerator.device,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids
        
    def compute_log_prob(self, latents, noise, prompt_embeds, pooled_prompt_embeds, 
                        text_ids, timesteps, noise_scheduler, with_grad=False):
        """Compute log probabilities for FLUX."""
        return pipeline_with_logprob(
            self.pipeline,
            latents=latents,
            noise=noise,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, 
            text_ids=text_ids,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            with_grad=with_grad
        )
        
    def sample_sde_step(self, latents, noise, prompt_embeds, pooled_prompt_embeds,
                       text_ids, timesteps, noise_scheduler, noise_level=0.1):
        """Sample using SDE step for FLUX."""
        return sde_step_with_logprob(
            self.pipeline,
            latents=latents,
            noise=noise,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids, 
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            noise_level=noise_level
        )