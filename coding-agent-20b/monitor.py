"""
Production Training Monitor

Features:
- Real-time loss spike detection
- Gradient norm monitoring
- Learning rate scheduling visualization
- Throughput tracking
- Checkpoint validation
- Automatic recovery from failures
- WandB/TensorBoard integration
"""

import os
import json
import time
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single step."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_per_second: float
    tokens_processed: int
    timestamp: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LossSpikeDetector:
    """Detect and handle loss spikes during training."""
    
    def __init__(
        self,
        window_size: int = 100,
        spike_threshold: float = 3.0,  # 3x standard deviation
        recovery_threshold: float = 0.5,
    ):
        """
        Args:
            window_size: Number of recent losses to track
            spike_threshold: Number of std devs above mean to consider a spike
            recovery_threshold: Fraction of spike that must recover to continue
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.recovery_threshold = recovery_threshold
        
        self.losses = deque(maxlen=window_size)
        self.spike_detected = False
        self.spike_step = None
        self.spike_loss = None
    
    def update(self, step: int, loss: float) -> Dict[str, any]:
        """
        Update with new loss and check for spikes.
        
        Returns:
            Dict with spike_detected, should_checkpoint, should_reduce_lr
        """
        self.losses.append(loss)
        
        result = {
            'spike_detected': False,
            'should_checkpoint': False,
            'should_reduce_lr': False,
            'spike_info': None,
        }
        
        # Need enough data to detect spikes
        if len(self.losses) < 10:
            return result
        
        # Calculate statistics
        mean_loss = np.mean(self.losses)
        std_loss = np.std(self.losses)
        
        # Check for spike
        if not self.spike_detected:
            if loss > mean_loss + self.spike_threshold * std_loss:
                self.spike_detected = True
                self.spike_step = step
                self.spike_loss = loss
                
                result['spike_detected'] = True
                result['should_checkpoint'] = True
                result['should_reduce_lr'] = True
                result['spike_info'] = {
                    'step': step,
                    'loss': loss,
                    'mean': mean_loss,
                    'std': std_loss,
                    'deviation': (loss - mean_loss) / std_loss,
                }
                
                logger.warning(
                    f"Loss spike detected at step {step}: "
                    f"loss={loss:.4f}, mean={mean_loss:.4f}, "
                    f"std={std_loss:.4f}, deviation={result['spike_info']['deviation']:.2f}σ"
                )
        else:
            # Check for recovery
            recovery = (self.spike_loss - loss) / (self.spike_loss - mean_loss)
            if recovery > self.recovery_threshold:
                logger.info(
                    f"Loss recovered at step {step}: "
                    f"recovery={recovery:.2%}"
                )
                self.spike_detected = False
                self.spike_step = None
                self.spike_loss = None
        
        return result


class GradientMonitor:
    """Monitor gradient norms and detect anomalies."""
    
    def __init__(self, window_size: int = 100, explosion_threshold: float = 10.0):
        self.window_size = window_size
        self.explosion_threshold = explosion_threshold
        self.grad_norms = deque(maxlen=window_size)
    
    def update(self, step: int, grad_norm: float) -> Dict[str, any]:
        """
        Update with new gradient norm.
        
        Returns:
            Dict with explosion_detected, should_clip, stats
        """
        self.grad_norms.append(grad_norm)
        
        result = {
            'explosion_detected': False,
            'should_clip': False,
            'stats': None,
        }
        
        if len(self.grad_norms) < 10:
            return result
        
        mean_norm = np.mean(self.grad_norms)
        max_norm = np.max(self.grad_norms)
        
        # Check for gradient explosion
        if grad_norm > self.explosion_threshold * mean_norm:
            result['explosion_detected'] = True
            result['should_clip'] = True
            
            logger.warning(
                f"Gradient explosion at step {step}: "
                f"norm={grad_norm:.4f}, mean={mean_norm:.4f}, "
                f"ratio={grad_norm/mean_norm:.2f}x"
            )
        
        result['stats'] = {
            'mean': mean_norm,
            'max': max_norm,
            'current': grad_norm,
        }
        
        return result


class ThroughputTracker:
    """Track training throughput and detect slowdowns."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.throughputs = deque(maxlen=window_size)
    
    def update(self, tokens_per_second: float) -> Dict[str, float]:
        """Update with new throughput measurement."""
        self.throughputs.append(tokens_per_second)
        
        if len(self.throughputs) < 10:
            return {'current': tokens_per_second}
        
        mean_throughput = np.mean(self.throughputs)
        
        return {
            'current': tokens_per_second,
            'mean': mean_throughput,
            'min': np.min(self.throughputs),
            'max': np.max(self.throughputs),
            'efficiency': tokens_per_second / mean_throughput if mean_throughput > 0 else 0,
        }


class CheckpointValidator:
    """Validate checkpoints by loading and testing them."""
    
    def __init__(self, validation_data, model_class, config):
        self.validation_data = validation_data
        self.model_class = model_class
        self.config = config
    
    def validate(self, checkpoint_path: str) -> Dict[str, any]:
        """
        Validate a checkpoint.
        
        Returns:
            Dict with valid, loss, error
        """
        import torch
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model and load state
            model = self.model_class(self.config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Run validation
            model.eval()
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in self.validation_data:
                    output = model(**batch)
                    total_loss += output.loss.item()
                    num_batches += 1
                    
                    if num_batches >= 10:  # Quick validation
                        break
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # Check for NaN/Inf
            if math.isnan(avg_loss) or math.isinf(avg_loss):
                return {
                    'valid': False,
                    'loss': avg_loss,
                    'error': 'Loss is NaN or Inf',
                }
            
            return {
                'valid': True,
                'loss': avg_loss,
                'error': None,
            }
        
        except Exception as e:
            return {
                'valid': False,
                'loss': None,
                'error': str(e),
            }


class TrainingMonitor:
    """Main training monitor that integrates all components."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        project_name: str = "coding-agent-20b",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loss_detector = LossSpikeDetector()
        self.grad_monitor = GradientMonitor()
        self.throughput_tracker = ThroughputTracker()
        
        # Initialize loggers
        self.use_wandb = use_wandb and HAS_WANDB
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        
        if self.use_wandb:
            try:
                wandb.init(project=project_name, config={})
                logger.info("WandB initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                logger.info("TensorBoard initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False
        
        # Metrics history
        self.metrics_history = []
    
    def log_step(self, metrics: TrainingMetrics) -> Dict[str, any]:
        """
        Log a training step and check for issues.
        
        Returns:
            Dict with recommendations (checkpoint, reduce_lr, etc.)
        """
        # Update components
        loss_result = self.loss_detector.update(metrics.step, metrics.loss)
        grad_result = self.grad_monitor.update(metrics.step, metrics.grad_norm)
        throughput_result = self.throughput_tracker.update(metrics.tokens_per_second)
        
        # Store metrics
        self.metrics_history.append(metrics.to_dict())
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                'train/loss': metrics.loss,
                'train/learning_rate': metrics.learning_rate,
                'train/grad_norm': metrics.grad_norm,
                'train/tokens_per_second': metrics.tokens_per_second,
                'train/tokens_processed': metrics.tokens_processed,
                'step': metrics.step,
            })
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar('train/loss', metrics.loss, metrics.step)
            self.tb_writer.add_scalar('train/learning_rate', metrics.learning_rate, metrics.step)
            self.tb_writer.add_scalar('train/grad_norm', metrics.grad_norm, metrics.step)
            self.tb_writer.add_scalar('train/tokens_per_second', metrics.tokens_per_second, metrics.step)
        
        # Save metrics to file
        if metrics.step % 100 == 0:
            metrics_file = self.log_dir / "metrics.jsonl"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        
        # Build recommendations
        recommendations = {
            'should_checkpoint': loss_result['should_checkpoint'],
            'should_reduce_lr': loss_result['should_reduce_lr'],
            'should_clip_grads': grad_result['should_clip'],
            'spike_detected': loss_result['spike_detected'],
            'grad_explosion': grad_result['explosion_detected'],
        }
        
        return recommendations
    
    def log_summary(self, step: int):
        """Log summary statistics."""
        if not self.metrics_history:
            return
        
        recent = self.metrics_history[-100:]
        
        summary = {
            'step': step,
            'avg_loss': np.mean([m['loss'] for m in recent]),
            'avg_grad_norm': np.mean([m['grad_norm'] for m in recent]),
            'avg_throughput': np.mean([m['tokens_per_second'] for m in recent]),
            'total_tokens': self.metrics_history[-1]['tokens_processed'],
        }
        
        logger.info(
            f"Step {step}: "
            f"loss={summary['avg_loss']:.4f}, "
            f"grad_norm={summary['avg_grad_norm']:.4f}, "
            f"throughput={summary['avg_throughput']:.0f} tok/s, "
            f"tokens={summary['total_tokens']/1e9:.2f}B"
        )
        
        if self.use_wandb:
            wandb.log({f'summary/{k}': v for k, v in summary.items()})
    
    def close(self):
        """Close all loggers."""
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.tb_writer.close()


class RecoveryManager:
    """Manage training recovery from failures."""
    
    def __init__(self, checkpoint_dir: str, max_retries: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_retries = max_retries
        self.retry_count = 0
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest valid checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        for ckpt in reversed(checkpoints):
            # Quick validation
            try:
                import torch
                checkpoint = torch.load(ckpt, map_location='cpu')
                if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                    return str(ckpt)
            except Exception:
                continue
        
        return None
    
    def should_retry(self) -> bool:
        """Check if we should retry after failure."""
        return self.retry_count < self.max_retries
    
    def record_retry(self):
        """Record a retry attempt."""
        self.retry_count += 1
    
    def reset_retries(self):
        """Reset retry counter after successful step."""
        self.retry_count = 0


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    monitor = TrainingMonitor(log_dir="test_logs")
    
    # Simulate training
    for step in range(100):
        metrics = TrainingMetrics(
            step=step,
            loss=2.0 + np.random.randn() * 0.1,
            learning_rate=1e-4,
            grad_norm=1.0 + np.random.randn() * 0.1,
            tokens_per_second=10000 + np.random.randn() * 1000,
            tokens_processed=step * 8192 * 32,
            timestamp=time.time(),
        )
        
        # Simulate loss spike
        if step == 50:
            metrics.loss = 10.0
        
        recommendations = monitor.log_step(metrics)
        
        if recommendations['spike_detected']:
            logger.info("Spike detected! Should checkpoint and reduce LR")
        
        if step % 10 == 0:
            monitor.log_summary(step)
    
    monitor.close()
    logger.info("Training monitor test complete")
