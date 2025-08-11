#!/usr/bin/env python3
"""
Refactored SD3 GRPO training script using the new trainer architecture.
"""

from absl import app, flags
from ml_collections import config_flags
from flow_grpo.training.trainers import SD3GRPOTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")


def main(_):
    """Main training function."""
    config = FLAGS.config
    trainer = SD3GRPOTrainer(config)
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    app.run(main)