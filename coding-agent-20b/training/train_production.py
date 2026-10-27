"""
Production Training Script with Monitoring

Integrates:
- DeepSpeed ZeRO-3
- Mixed precision (BF16/FP16)
- Activation checkpointing
- Real-time monitoring (loss spikes, grad norms, throughput)
- Automatic recovery from failures
- Checkpoint validation

Usage:
    deepspeed --num_gpus=8 training/train_production.py \\
        --config configs/training_config.json \\
        --data_dir data/processed \\
        --output_dir checkpoints
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import CodingAgentConfig
from model.model import CodingAgentModel
from dataset.processor import StreamingDataset
from training.monitor import (
    TrainingMonitor, TrainingMetrics, RecoveryManager, RecoveryPolicy
)
from training.checkpoint import save_checkpoint_atomic, latest_valid_checkpoint
from training.schedules import seq_length_at, wsd_lr

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("WARNING: DeepSpeed not installed")

logger = logging.getLogger(__name__)


def setup_logging(rank: int, log_dir: str):
    """Setup logging."""
    log_file = Path(log_dir) / f"train_rank{rank}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if rank == 0 else logging.NullHandler(),
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path) as f:
        return json.load(f)


def create_model(config: CodingAgentConfig) -> CodingAgentModel:
    """Create and initialize model."""
    model = CodingAgentModel(config)
    
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def create_dataloader(
    data_dir: str,
    batch_size: int,
    seq_length: int,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create data loader."""
    dataset = StreamingDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        rank=rank,
        world_size=world_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming dataset handles shuffling
        num_workers=4,
        pin_memory=True,
    )


def train(args: argparse.Namespace):
    """Main training function."""
    # Initialize distributed
    if HAS_DEEPSPEED:
        deepspeed.init_distributed()
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Setup logging
    setup_logging(rank, args.log_dir)
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Production Training Script")
        logger.info("=" * 60)
    
    # Load config
    config_dict = load_config(args.config)
    model_config = CodingAgentConfig.from_dict(config_dict['model'])
    train_config = config_dict['training']
    
    if rank == 0:
        model_config.print_summary()
        logger.info(f"World size: {world_size} GPUs")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(model_config)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=train_config['batch_size'],
        seq_length=model_config.max_seq_length,
        rank=rank,
        world_size=world_size,
    )
    
    # Initialize DeepSpeed
    if HAS_DEEPSPEED:
        logger.info("Initializing DeepSpeed...")
        ds_config_path = Path(args.config).parent / "ds_config_zero3.json"
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=str(ds_config_path),
        )
    else:
        # Fallback to standard PyTorch
        logger.warning("DeepSpeed not available, using standard training")
        model_engine = model.to(local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
        scheduler = None
    
    # Initialize monitor (rank 0 only)
    monitor = None
    if rank == 0:
        monitor = TrainingMonitor(
            log_dir=args.log_dir,
            project_name=args.project_name,
            use_wandb=args.use_wandb,
            use_tensorboard=True,
        )
    
    recovery_policy = RecoveryPolicy()
    # Initialize recovery manager
    recovery = RecoveryManager(
        checkpoint_dir=args.output_dir,
        max_retries=3,
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    tokens_processed = 0
    start_time = time.time()
    
    for epoch in range(train_config['num_epochs']):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            
            # Move to device
            input_ids = batch['input_ids'].to(model_engine.device)
            labels = batch['labels'].to(model_engine.device)
            
            # Forward pass
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            if HAS_DEEPSPEED:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            
            # Calculate metrics
            batch_time = time.time() - batch_start
            tokens_in_batch = input_ids.numel()
            tokens_per_second = tokens_in_batch / batch_time
            
            global_step += 1
            tokens_processed += tokens_in_batch
            
            # Get gradient norm
            grad_norm = 0.0
            if hasattr(model_engine, 'get_global_grad_norm'):
                grad_norm = model_engine.get_global_grad_norm()
            elif hasattr(optimizer, 'param_groups'):
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5
            
            # Get learning rate
            lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else train_config['learning_rate']
            
            # Log metrics (rank 0 only)
            if rank == 0 and monitor:
                metrics = TrainingMetrics(
                    step=global_step,
                    loss=loss.item(),
                    learning_rate=lr,
                    grad_norm=grad_norm,
                    tokens_per_second=tokens_per_second,
                    tokens_processed=tokens_processed,
                    timestamp=time.time(),
                )
                
                recommendations = monitor.log_step(metrics)
                
                action = recovery_policy.decide(global_step, loss.item(), recommendations)
                if action == "skip_batch":
                    logger.warning(f"Skipping batch at step {global_step} (policy: {action})")
                    continue
                if action == "rollback":
                    latest = latest_valid_checkpoint(args.output_dir)
                    logger.warning(f"Rolling back to verified checkpoint: {latest}")
                    if latest:
                        from training.checkpoint import load_verified
                        state = load_verified(latest)
                        model_engine.load_state_dict(state['model_state_dict'])
                if action == "reduce_lr":
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5

                # Handle recommendations
                if recommendations['should_checkpoint']:
                    logger.info("Loss spike detected - saving checkpoint")
                    save_checkpoint(
                        model_engine, optimizer, global_step,
                        args.output_dir, rank
                    )
                
                if recommendations['should_reduce_lr']:
                    logger.info("Reducing learning rate due to loss spike")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                
                # Log summary periodically
                if global_step % 100 == 0:
                    monitor.log_summary(global_step)
            
            # Save checkpoint periodically
            if global_step % train_config['checkpoint_steps'] == 0:
                save_checkpoint(
                    model_engine, optimizer, global_step,
                    args.output_dir, rank
                )
                
                if rank == 0:
                    recovery.reset_retries()
            
            # Log progress
            if rank == 0 and global_step % 10 == 0:
                elapsed = time.time() - start_time
                avg_tokens_per_sec = tokens_processed / elapsed
                
                logger.info(
                    f"Step {global_step}: "
                    f"loss={loss.item():.4f}, "
                    f"lr={lr:.2e}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"tokens/s={tokens_per_second:.0f}, "
                    f"total_tokens={tokens_processed/1e9:.2f}B"
                )
    
    # Final checkpoint
    save_checkpoint(
        model_engine, optimizer, global_step,
        args.output_dir, rank
    )
    
    if rank == 0:
        logger.info("Training complete!")
        if monitor:
            monitor.close()


def save_checkpoint(
    model_engine,
    optimizer,
    step: int,
    output_dir: str,
    rank: int,
):
    """Save training checkpoint."""
    if rank != 0:
        return
    
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
    
    if HAS_DEEPSPEED and hasattr(model_engine, 'save_checkpoint'):
        model_engine.save_checkpoint(str(checkpoint_dir), tag=f"step_{step}")
    else:
        save_checkpoint_atomic({
            'step': step,
            'model_state_dict': model_engine.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(checkpoint_dir), step, keep_last=5)
        return
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Keep only last 5 checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) > 5:
        for old_ckpt in checkpoints[:-5]:
            old_ckpt.unlink()


def main():
    parser = argparse.ArgumentParser(description="Production training")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--project_name", type=str, default="coding-agent-20b")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB")
    
    if HAS_DEEPSPEED:
        parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
