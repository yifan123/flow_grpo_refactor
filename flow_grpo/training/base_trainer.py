import os
import datetime
import tempfile
import time
import json
import hashlib
from abc import ABC, abstractmethod
from functools import partial

import torch
import wandb
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from PIL import Image
import tqdm

import flow_grpo.prompts
from flow_grpo.reward import multi_score
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ema import EMAModuleWrapper
from .datasets import DistributedKRepeatSampler
from .utils import calculate_zero_std_ratio, unwrap_model, save_ckpt, create_generator, load_dataset_class

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Base trainer class for all Flow-GRPO training algorithms."""
    
    def __init__(self, config):
        self.config = config
        self.setup_config()
        self.setup_accelerator()
        self.setup_logging()
        
    def setup_config(self):
        """Setup basic configuration and unique identifiers."""
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if self.config.run_name == "":
            self.config.run_name = unique_id
        else:
            self.config.run_name += "_" + unique_id
            
    def setup_accelerator(self):
        """Initialize the Accelerator with proper configuration."""
        logging_dir = os.path.join(self.config.logdir, self.config.run_name)
        
        config = ProjectConfiguration(
            project_dir=self.config.logdir,
            logging_dir=logging_dir,
        )
        
        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=config,
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            split_batches=self.config.train.split_batches,
        )
        
        if self.accelerator.is_main_process:
            os.makedirs(logging_dir, exist_ok=True)
            
        self.log_dir = logging_dir
        
    def setup_logging(self):
        """Setup wandb logging."""
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config.tracker_project_name,
                config=dict(self.config),
                init_kwargs={"wandb": {"name": self.config.run_name}},
            )
            
    @abstractmethod
    def load_pipeline(self):
        """Load the diffusion pipeline. Must be implemented by subclasses."""
        pass
        
    @abstractmethod  
    def load_model_components(self):
        """Load and setup model components (VAE, text encoders, etc). Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def compute_text_embeddings(self, prompts):
        """Compute text embeddings. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def compute_log_prob(self, *args, **kwargs):
        """Compute log probabilities. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute training loss. Must be implemented by subclasses."""
        pass
        
    def setup_model(self):
        """Setup the complete model pipeline."""
        # Load pipeline and components
        self.pipeline = self.load_pipeline()
        self.load_model_components()
        
        # Setup LoRA
        self.setup_lora()
        
        # Move models to appropriate devices
        self.setup_model_devices()
        
    def setup_lora(self):
        """Setup LoRA configuration and adapters."""
        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=self.get_lora_target_modules(),
            lora_dropout=self.config.lora.dropout,
        )
        
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer = get_peft_model(transformer, lora_config)
        setattr(self.pipeline, self.get_transformer_attr_name(), transformer)
        
        # Load checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint()
            
    @abstractmethod
    def get_lora_target_modules(self):
        """Get LoRA target modules. Must be implemented by subclasses."""
        pass
        
    @abstractmethod  
    def get_transformer_attr_name(self):
        """Get transformer attribute name in pipeline. Must be implemented by subclasses."""
        pass
        
    def setup_model_devices(self):
        """Move model components to appropriate devices."""
        if hasattr(self.pipeline, 'vae'):
            self.pipeline.vae.to(self.accelerator.device, dtype=torch.float32)
        
        # Text encoders setup - implemented by subclasses
        self.setup_text_encoders()
        
        # Transformer setup
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer.to(self.accelerator.device)
        
    @abstractmethod
    def setup_text_encoders(self):
        """Setup text encoders on appropriate devices. Must be implemented by subclasses."""
        pass
        
    def setup_optimizer(self):
        """Setup the optimizer."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        
        if self.config.train.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("bitsandbytes not available for 8bit Adam")
        else:
            optimizer_cls = torch.optim.AdamW
            
        self.optimizer = optimizer_cls(
            transformer.parameters(),
            lr=self.config.train.learning_rate,
            betas=(self.config.train.adam_beta1, self.config.train.adam_beta2),
            weight_decay=self.config.train.adam_weight_decay,
            eps=self.config.train.adam_epsilon,
        )
        
    def setup_data_loader(self):
        """Setup the data loader."""
        dataset_class = load_dataset_class(
            self.config.sample.dataset, 
            self.get_dataset_class_name()
        )
        dataset = dataset_class(self.config.sample.dataset)
        
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            batch_size=self.config.sample.train_batch_size,
            k=self.config.sample.num_images_per_prompt,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            seed=self.config.sample.seed,
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.sample.train_batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=True,
        )
        
    def get_dataset_class_name(self):
        """Get dataset class name. Can be overridden by subclasses."""
        return "TextPromptDataset"
        
    def prepare_for_training(self):
        """Prepare models and optimizer for distributed training."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        
        # Apply EMA if configured
        if self.config.train.ema_decay != 1.0:
            transformer = EMAModuleWrapper(transformer, self.config.train.ema_decay)
            setattr(self.pipeline, self.get_transformer_attr_name(), transformer)
        
        # Prepare with accelerator
        transformer, self.optimizer, self.dataloader = self.accelerator.prepare(
            transformer, self.optimizer, self.dataloader
        )
        setattr(self.pipeline, self.get_transformer_attr_name(), transformer)
        
        # Compile if configured
        if self.config.train.compile:
            transformer = torch.compile(transformer, mode="reduce-overhead", fullgraph=True)
            setattr(self.pipeline, self.get_transformer_attr_name(), transformer)
            
    def load_checkpoint(self):
        """Load checkpoint if specified."""
        if not self.config.resume_from_checkpoint:
            return
            
        lora_state_dict = torch.load(self.config.resume_from_checkpoint, map_location="cpu")["model_state_dict"]
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        set_peft_model_state_dict(transformer, lora_state_dict)
        
        if self.accelerator.is_main_process:
            logger.info(f"Loaded checkpoint from {self.config.resume_from_checkpoint}")
            
    def setup_stat_tracking(self):
        """Setup statistics tracking."""
        self.stat_tracker = PerPromptStatTracker(
            buffer_size=self.config.stat_tracking.buffer_size,
            min_count=self.config.stat_tracking.min_count,
        )
        
    def train(self):
        """Main training loop."""
        set_seed(self.config.seed, device_specific=True)
        
        # Setup all components
        self.setup_model()
        self.setup_optimizer() 
        self.setup_data_loader()
        self.setup_stat_tracking()
        self.prepare_for_training()
        
        # Initialize rewards
        reward_fn = multi_score(self.accelerator.device, self.config.reward_fn)
        
        # Main training loop
        global_step = 0
        while global_step < self.config.train.max_train_steps:
            
            # Evaluation phase
            if global_step % self.config.sample.num_batches_per_epoch == 0:
                if self.accelerator.is_main_process:
                    eval_result = self.eval()
                    self.accelerator.log(eval_result, step=global_step)
                    
            # Sampling phase  
            info = self.sample_batch(reward_fn)
            
            # Training phase
            loss_info = self.training_step(info)
            
            # Logging
            self.log_training_metrics(info, loss_info, global_step)
            
            # Save checkpoint
            if (global_step + 1) % self.config.save_freq == 0 and self.accelerator.is_main_process:
                transformer = getattr(self.pipeline, self.get_transformer_attr_name())
                save_ckpt(
                    f"checkpoint_{global_step+1}.pth", 
                    self.log_dir, 
                    transformer, 
                    self.accelerator, 
                    self.config
                )
                
            global_step += 1
            
    @abstractmethod
    def sample_batch(self, reward_fn):
        """Sample a batch of data. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def training_step(self, info):
        """Perform one training step. Must be implemented by subclasses."""
        pass
        
    def eval(self):
        """Evaluation phase."""
        transformer = getattr(self.pipeline, self.get_transformer_attr_name())
        transformer.eval()
        
        eval_prompts = getattr(flow_grpo.prompts, self.config.sample.eval_prompts)
        
        with torch.no_grad():
            latents = torch.randn(
                (len(eval_prompts), 16, 128, 128),
                device=self.accelerator.device,
                dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32,
                generator=create_generator(self.config, len(eval_prompts))
            )
            
            images = self.pipeline(
                eval_prompts,
                num_inference_steps=self.config.sample.eval_num_inference_steps,
                guidance_scale=self.config.sample.eval_guidance_scale,
                latents=latents,
            ).images
            
        transformer.train()
        
        # Log images to wandb
        images_wandb = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                images_wandb.append(wandb.Image(tmp_file.name, caption=eval_prompts[i]))
                
        return {"eval_images": images_wandb}
        
    def log_training_metrics(self, info, loss_info, global_step):
        """Log training metrics."""
        # Convert lists to tensors and compute means
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], torch.Tensor):
                        info[key] = torch.mean(torch.stack(value))
                    else:
                        info[key] = np.mean(value)
        
        # Add loss info
        info.update(loss_info)
        info["global_step"] = global_step
        
        # Reduce across processes
        info = self.accelerator.reduce(info, reduction="mean")
        
        # Log to wandb
        if self.accelerator.is_main_process:
            wandb.log(info, step=global_step)
        
    def cleanup(self):
        """Cleanup resources."""
        self.accelerator.end_training()