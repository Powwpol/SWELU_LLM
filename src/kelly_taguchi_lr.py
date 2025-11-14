"""
Kelly-Taguchi Learning Rate Scheduler

Adapts learning rate dynamically based on loss progression.
Combines Kelly criterion with Taguchi's quality engineering principles.
"""

import torch
import math
from typing import Optional, List


class KellyTaguchiLR:
    """
    Kelly-Taguchi adaptive learning rate scheduler.
    
    Principle:
    - Increases LR when loss is decreasing consistently (Kelly: bet more when winning)
    - Decreases LR when loss plateaus or increases (Taguchi: reduce variation)
    - Uses moving average to smooth decisions
    
    Args:
        optimizer: PyTorch optimizer
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        window_size: Window for moving average of loss
        increase_factor: Factor to increase LR when improving
        decrease_factor: Factor to decrease LR when stagnating
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 3e-4,
        warmup_steps: int = 1000,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        window_size: int = 100,
        increase_factor: float = 1.05,
        decrease_factor: float = 0.95,
        patience: int = 10,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.window_size = window_size
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience
        
        self.current_lr = base_lr
        self.step_count = 0
        self.loss_history: List[float] = []
        self.no_improvement_count = 0
        self.best_avg_loss = float('inf')
        
    def get_warmup_lr(self, step: int) -> float:
        """Linear warmup learning rate."""
        return self.base_lr * (step / self.warmup_steps)
    
    def get_moving_avg_loss(self) -> Optional[float]:
        """Calculate moving average of recent losses."""
        if len(self.loss_history) < self.window_size:
            if len(self.loss_history) == 0:
                return None
            return sum(self.loss_history) / len(self.loss_history)
        
        recent_losses = self.loss_history[-self.window_size:]
        return sum(recent_losses) / len(recent_losses)
    
    def step(self, loss: float):
        """
        Update learning rate based on loss.
        
        Args:
            loss: Current training loss
        """
        self.step_count += 1
        self.loss_history.append(loss)
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            new_lr = self.get_warmup_lr(self.step_count)
        else:
            # Adaptive phase
            avg_loss = self.get_moving_avg_loss()
            
            if avg_loss is None:
                new_lr = self.current_lr
            else:
                # Kelly-Taguchi logic
                if avg_loss < self.best_avg_loss:
                    # Loss is improving - increase LR (Kelly: bet more when winning)
                    new_lr = self.current_lr * self.increase_factor
                    self.best_avg_loss = avg_loss
                    self.no_improvement_count = 0
                else:
                    # Loss is stagnating or increasing
                    self.no_improvement_count += 1
                    
                    if self.no_improvement_count >= self.patience:
                        # Decrease LR (Taguchi: reduce variation)
                        new_lr = self.current_lr * self.decrease_factor
                        self.no_improvement_count = 0
                    else:
                        new_lr = self.current_lr
                
                # Clip to bounds
                new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        # Update optimizer
        self.current_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'step_count': self.step_count,
            'current_lr': self.current_lr,
            'loss_history': self.loss_history,
            'best_avg_loss': self.best_avg_loss,
            'no_improvement_count': self.no_improvement_count,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.step_count = state_dict.get('step_count', 0)
        self.current_lr = state_dict.get('current_lr', self.base_lr)
        self.loss_history = state_dict.get('loss_history', [])
        self.best_avg_loss = state_dict.get('best_avg_loss', float('inf'))
        self.no_improvement_count = state_dict.get('no_improvement_count', 0)


if __name__ == "__main__":
    """Test Kelly-Taguchi scheduler."""
    import torch.optim as optim
    
    print("=" * 80)
    print("🧪 TESTING KELLY-TAGUCHI LR SCHEDULER")
    print("=" * 80)
    print()
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    # Create scheduler
    scheduler = KellyTaguchiLR(
        optimizer=optimizer,
        base_lr=3e-4,
        warmup_steps=100,
        max_lr=1e-3,
        min_lr=1e-5,
        window_size=20,
    )
    
    # Simulate training with decreasing loss
    print("Phase 1: Warmup (steps 0-100)")
    for step in range(100):
        loss = 10.0 - step * 0.05  # Decreasing loss
        scheduler.step(loss)
        if step % 20 == 0:
            print(f"  Step {step}: LR = {scheduler.get_last_lr():.6f}, Loss = {loss:.4f}")
    
    print("\nPhase 2: Improving phase (steps 100-200)")
    for step in range(100, 200):
        loss = 5.0 - (step - 100) * 0.01  # Continuing to decrease
        scheduler.step(loss)
        if step % 20 == 0:
            print(f"  Step {step}: LR = {scheduler.get_last_lr():.6f}, Loss = {loss:.4f}")
    
    print("\nPhase 3: Plateau (steps 200-300)")
    for step in range(200, 300):
        loss = 4.0 + torch.randn(1).item() * 0.1  # Plateau with noise
        scheduler.step(loss)
        if step % 20 == 0:
            print(f"  Step {step}: LR = {scheduler.get_last_lr():.6f}, Loss = {loss:.4f}")
    
    print("\n✅ Kelly-Taguchi LR Scheduler Test Complete!")
    print(f"Final LR: {scheduler.get_last_lr():.6f}")

