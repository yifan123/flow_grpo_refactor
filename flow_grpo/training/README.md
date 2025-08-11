# Flow-GRPO Refactored Training Architecture

This module contains the refactored training architecture that eliminates code duplication across the original training scripts.

## Architecture Overview

### Key Components

1. **Base Trainer** (`base_trainer.py`): Abstract base class with common training logic
2. **Model-Specific Trainers**: Implement model-specific details (SD3, FLUX, FLUX-Kontext)
3. **Algorithm Mixins** (`algorithms.py`): Implement training algorithms (GRPO, DPO, SFT)
4. **Shared Utilities** (`datasets.py`, `utils.py`): Common dataset classes and utility functions
5. **Concrete Trainers** (`trainers.py`): Combine model trainers with algorithm mixins


## Usage

### Using Refactored Scripts

Replace the original scripts with refactored versions:

```bash
# Original
python scripts/train_sd3.py --config=config/grpo.py

# Refactored (same functionality)
python scripts/train_sd3_refactored.py --config=config/grpo.py
```

### Creating Custom Trainers

```python
from flow_grpo.training.sd3_trainer import SD3Trainer
from flow_grpo.training.algorithms import GRPOMixin

class MyCustomTrainer(SD3Trainer, GRPOMixin):
    def custom_method(self):
        # Your custom logic here
        pass

# Use the trainer
trainer = MyCustomTrainer(config)
trainer.train()
```

### Adding New Models

1. Create a new model trainer in `flow_grpo/training/`:

```python
# new_model_trainer.py
from .base_trainer import BaseTrainer

class NewModelTrainer(BaseTrainer):
    def load_pipeline(self):
        # Load your model pipeline
        pass
        
    def compute_text_embeddings(self, prompts):
        # Implement text embedding computation
        pass
        
    # Implement other abstract methods...
```

2. Add combinations in `trainers.py`:

```python
class NewModelGRPOTrainer(NewModelTrainer, GRPOMixin):
    pass
```

### Adding New Algorithms

1. Create a new algorithm mixin in `algorithms.py`:

```python
class NewAlgorithmMixin:
    def sample_batch(self, reward_fn):
        # Implement sampling logic
        pass
        
    def training_step(self, info):
        # Implement training step
        pass
```

2. Combine with existing model trainers:

```python
class SD3NewAlgorithmTrainer(SD3Trainer, NewAlgorithmMixin):
    pass
```

## Class Hierarchy

```
BaseTrainer (Abstract)
├── SD3Trainer
│   ├── SD3GRPOTrainer (+ GRPOMixin)
│   ├── SD3DPOTrainer (+ DPOMixin) 
│   ├── SD3SFTTrainer (+ SFTMixin)
│   └── SD3S1GRPOTrainer (+ S1Mixin + GRPOMixin)
├── FluxTrainer  
│   └── FluxGRPOTrainer (+ GRPOMixin)
└── FluxKontextTrainer
    └── FluxKontextGRPOTrainer (+ GRPOMixin)
```

## Migration Guide

The refactored scripts maintain full compatibility with existing configurations and command-line interfaces. No changes needed to:

- Configuration files (`config/*.py`)
- Shell scripts (`scripts/single_node/*.sh`, `scripts/multi_node/*/*.sh`)
- Command-line arguments
- Output formats and checkpoints


## Files Overview

- `base_trainer.py`: Abstract base trainer with template methods
- `datasets.py`: Shared dataset classes (TextPromptDataset, GenevalPromptDataset, etc.)
- `utils.py`: Common utility functions (save_ckpt, unwrap_model, etc.)
- `sd3_trainer.py`: SD3-specific model trainer
- `flux_trainer.py`: FLUX-specific model trainer  
- `flux_kontext_trainer.py`: FLUX-Kontext-specific model trainer
- `algorithms.py`: Algorithm mixins (GRPOMixin, DPOMixin, SFTMixin)
- `trainers.py`: Concrete trainer combinations
- `__init__.py`: Package initialization