"""
Concrete trainer implementations that combine model-specific trainers with algorithm mixins.
"""

from .sd3_trainer import SD3Trainer
from .flux_trainer import FluxTrainer  
from .flux_kontext_trainer import FluxKontextTrainer
from .algorithms import GRPOMixin, DPOMixin, SFTMixin


class SD3GRPOTrainer(SD3Trainer, GRPOMixin):
    """SD3 trainer with GRPO algorithm."""
    pass


class SD3DPOTrainer(SD3Trainer, DPOMixin):
    """SD3 trainer with DPO algorithm.""" 
    pass


class SD3SFTTrainer(SD3Trainer, SFTMixin):
    """SD3 trainer with SFT algorithm."""
    pass


class FluxGRPOTrainer(FluxTrainer, GRPOMixin):
    """FLUX trainer with GRPO algorithm."""
    pass


class FluxKontextGRPOTrainer(FluxKontextTrainer, GRPOMixin):
    """FLUX Kontext trainer with GRPO algorithm.""" 
    pass


# For the S1 variant, we need a special mixin
class S1Mixin:
    """S1 (Single Step) variant mixin."""
    
    def load_pipeline_components(self):
        """Load S1-specific pipeline components."""
        # Import S1-specific modules
        from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_s1 import pipeline_with_logprob
        from flow_grpo.diffusers_patch.sd3_sde_with_logprob_s1 import sde_step_with_logprob
        
        self._pipeline_with_logprob = pipeline_with_logprob
        self._sde_step_with_logprob = sde_step_with_logprob
        
    def compute_log_prob(self, *args, **kwargs):
        """Use S1-specific log prob computation."""
        return self._pipeline_with_logprob(self.pipeline, *args, **kwargs)
        
    def sample_sde_step(self, *args, **kwargs):
        """Use S1-specific SDE step."""
        return self._sde_step_with_logprob(self.pipeline, *args, **kwargs)
        
    def _generate_samples_grpo(self, prompts, embeddings, reward_fn):
        """S1-specific sample generation with mini batching."""
        info = super()._generate_samples_grpo(prompts, embeddings, reward_fn)
        
        # S1-specific logic for mini_num_image_per_prompt
        if hasattr(self.config.sample, 'mini_num_image_per_prompt'):
            # Process samples in mini-batches for memory efficiency
            mini_size = self.config.sample.mini_num_image_per_prompt
            # Implementation would batch process the samples
            pass
            
        return info


class SD3S1GRPOTrainer(SD3Trainer, S1Mixin, GRPOMixin):
    """SD3 trainer with S1 variant and GRPO algorithm."""
    
    def setup_model(self):
        """Setup model with S1-specific components."""
        super().setup_model()
        self.load_pipeline_components()


# Export all trainer classes
__all__ = [
    'SD3GRPOTrainer',
    'SD3DPOTrainer', 
    'SD3SFTTrainer',
    'SD3S1GRPOTrainer',
    'FluxGRPOTrainer',
    'FluxKontextGRPOTrainer'
]