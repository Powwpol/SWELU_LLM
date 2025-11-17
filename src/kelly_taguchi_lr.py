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
        increase_factor: float = 1.0,
        decrease_factor: float = 1.0,
        patience: int = 10,  # conservé pour compat mais plus utilisé directement
        plateau_tol: float = 1e-3,
        max_adjust_fraction: float = 0.2,
        weibull_shape: float = 1.5,
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
        self.plateau_tol = plateau_tol
        self.max_adjust_fraction = max_adjust_fraction
        self.weibull_shape = weibull_shape
        
        self.current_lr = base_lr
        self.step_count = 0
        self.loss_history: List[float] = []
        self.no_improvement_count = 0  # gardé pour compatibilité
        self.best_avg_loss = float('inf')  # gardé pour compatibilité
        self._prev_avg_loss: Optional[float] = None  # plus utilisé directement

    def _weibull_median_and_prob(self, current_loss: float) -> Optional[tuple]:
        """
        Estime la médiane et la probabilité d'écart du loss courant
        sous une approximation de loi de Weibull ajustée sur la fenêtre récente.
        """
        window = self.loss_history[-self.window_size :] if self.loss_history else []
        if len(window) == 0:
            return None
        # Médiane empirique (robuste) comme estimateur de la médiane de Weibull
        sorted_win = sorted(window)
        n = len(sorted_win)
        if n % 2 == 1:
            median = sorted_win[n // 2]
        else:
            median = 0.5 * (sorted_win[n // 2 - 1] + sorted_win[n // 2])

        # Paramètre de forme fixé, on déduit l'échelle pour coller la médiane
        k = self.weibull_shape
        # median_theoretical = lambda * (ln 2)^(1/k)  -> lambda = median / (ln 2)^(1/k)
        if median <= 0:
            return None
        lam = median / (math.log(2.0) ** (1.0 / k))

        # CDF de Weibull: F(x) = 1 - exp(-(x/lambda)^k) pour x>=0
        x = max(current_loss, 1e-8)
        F_x = 1.0 - math.exp(-((x / lam) ** k))
        # Probabilité d'écart vs médiane (0.5) : distance en probabilité, bornée dans [0,1]
        p_dev = min(1.0, max(0.0, abs(F_x - 0.5) * 2.0))
        return median, p_dev
        
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
        Update learning rate based on loss using an inverted Kelly logic.
        
        - On plateau (little or no improvement), INCREASE LR by a Kelly-style fraction.
        - When loss clearly improves or degrades, DECREASE LR by a fraction.
        """
        self.step_count += 1
        self.loss_history.append(loss)
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            new_lr = self.get_warmup_lr(self.step_count)
        else:
            # Adaptive phase: utilise médiane + Weibull pour calibrer la fraction Kelly
            stats = self._weibull_median_and_prob(loss)
            if stats is None:
                new_lr = self.current_lr
            else:
                median, p_dev = stats
                # Fraction Kelly dimensionnée par la probabilité d'écart
                frac = min(self.max_adjust_fraction, max(0.0, p_dev))

                # Inversion logique demandée:
                # - Si loss proche de la médiane (plateau) -> on augmente LR d'une fraction
                # - Sinon (écart significatif) -> on diminue LR d'une fraction
                if abs(loss - median) <= self.plateau_tol:
                    adjust = 1.0 + self.increase_factor * frac
                else:
                    adjust = 1.0 - self.decrease_factor * frac
                    adjust = max(0.5, adjust)

                new_lr = self.current_lr * adjust
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

