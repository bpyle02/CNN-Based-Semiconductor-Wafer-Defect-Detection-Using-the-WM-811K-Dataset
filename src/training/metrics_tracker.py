"""
Training metrics tracker with moving averages and trend detection.

Inspired by RL training curve tracking where episode rewards are noisy
and moving averages reveal true learning progress. Adapted for supervised
learning metrics (loss, accuracy, learning rate).
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track training metrics with moving averages and trend detection.

    Features:
    - Raw metric storage per step/epoch
    - Configurable moving average window
    - Trend detection (improving, plateau, degrading)
    - Best metric tracking with patience
    - Export to JSON for analysis
    - Plotting with moving average overlay

    Example:
        tracker = MetricsTracker(window_size=5)
        for epoch in range(50):
            tracker.update('train_loss', loss_val, step=epoch)
            tracker.update('val_acc', acc_val, step=epoch)
        tracker.plot_all()
        tracker.to_json('metrics.json')
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Initialize metrics tracker.

        Args:
            window_size: Default window size for moving average computation.
                         Must be >= 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
        self.steps: Dict[str, List[int]] = {}
        self._auto_step: Dict[str, int] = {}

    def update(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric (e.g. 'train_loss', 'val_acc')
            value: Metric value to record
            step: Optional step/epoch number. If None, auto-increments.
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.steps[metric_name] = []
            self._auto_step[metric_name] = 0

        if step is None:
            step = self._auto_step[metric_name]

        self._auto_step[metric_name] = step + 1
        self.metrics[metric_name].append(float(value))
        self.steps[metric_name].append(step)

    def get_moving_average(self, metric_name: str, window: Optional[int] = None) -> List[float]:
        """
        Compute moving average of a metric.

        For the first (window - 1) values, the average uses all available
        values up to that point (expanding window).

        Args:
            metric_name: Name of the metric
            window: Window size. If None, uses self.window_size.

        Returns:
            List of moving average values, same length as raw values.

        Raises:
            KeyError: If metric_name has not been recorded.
        """
        if metric_name not in self.metrics:
            raise KeyError(f"No metric named '{metric_name}'")

        values = self.metrics[metric_name]
        if not values:
            return []

        w = window if window is not None else self.window_size
        w = max(1, w)

        result = []
        for i in range(len(values)):
            start = max(0, i - w + 1)
            window_vals = values[start : i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

    def get_trend(self, metric_name: str, lookback: int = 10) -> str:
        """
        Detect trend direction using linear regression slope on recent values.

        Args:
            metric_name: Name of the metric
            lookback: Number of recent values to consider

        Returns:
            One of 'improving', 'plateau', or 'degrading'.
            For loss-like metrics (name contains 'loss'), a negative slope
            means improving. For accuracy-like metrics, a positive slope
            means improving.

        Raises:
            KeyError: If metric_name has not been recorded.
        """
        if metric_name not in self.metrics:
            raise KeyError(f"No metric named '{metric_name}'")

        values = self.metrics[metric_name]
        if len(values) < 2:
            return "plateau"

        recent = values[-lookback:]
        n = len(recent)
        x = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)

        # Linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x == 0:
            return "plateau"
        slope = ((x - x_mean) * (y - y_mean)).sum() / var_x

        # Normalize slope relative to the mean value magnitude
        scale = abs(y_mean) if abs(y_mean) > 1e-10 else 1.0
        normalized_slope = slope / scale

        # Threshold for plateau detection
        threshold = 0.01

        is_loss = "loss" in metric_name.lower()

        if abs(normalized_slope) < threshold:
            return "plateau"
        elif is_loss:
            return "improving" if normalized_slope < 0 else "degrading"
        else:
            return "improving" if normalized_slope > 0 else "degrading"

    def get_best(self, metric_name: str, mode: str = "max") -> Tuple[float, int]:
        """
        Return the best value and its step for a metric.

        Args:
            metric_name: Name of the metric
            mode: 'max' for metrics where higher is better (accuracy),
                  'min' for metrics where lower is better (loss)

        Returns:
            Tuple of (best_value, best_step)

        Raises:
            KeyError: If metric_name has not been recorded.
            ValueError: If no values have been recorded for the metric.
        """
        if metric_name not in self.metrics:
            raise KeyError(f"No metric named '{metric_name}'")

        values = self.metrics[metric_name]
        steps = self.steps[metric_name]

        if not values:
            raise ValueError(f"No values recorded for '{metric_name}'")

        if mode == "min":
            idx = int(np.argmin(values))
        else:
            idx = int(np.argmax(values))

        return values[idx], steps[idx]

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Return summary of all tracked metrics.

        Returns:
            Dict mapping metric names to summary dicts containing:
            - last: Most recent value
            - best: Best value
            - best_step: Step of best value
            - moving_avg: Current moving average value
            - trend: Current trend direction
            - count: Number of recorded values
        """
        summary = {}
        for name in self.metrics:
            values = self.metrics[name]
            if not values:
                continue

            is_loss = "loss" in name.lower()
            mode = "min" if is_loss else "max"
            best_val, best_step = self.get_best(name, mode=mode)
            ma = self.get_moving_average(name)

            summary[name] = {
                "last": values[-1],
                "best": best_val,
                "best_step": best_step,
                "moving_avg": ma[-1] if ma else None,
                "trend": self.get_trend(name),
                "count": len(values),
            }
        return summary

    def to_json(self, path: str) -> None:
        """
        Export all metrics and summary to a JSON file.

        Args:
            path: File path for the JSON output
        """
        data = {
            "window_size": self.window_size,
            "metrics": {},
        }
        for name in self.metrics:
            ma = self.get_moving_average(name)
            is_loss = "loss" in name.lower()
            mode = "min" if is_loss else "max"
            best_val, best_step = (
                self.get_best(name, mode=mode) if self.metrics[name] else (None, None)
            )

            data["metrics"][name] = {
                "values": self.metrics[name],
                "steps": self.steps[name],
                "moving_average": ma,
                "best_value": best_val,
                "best_step": best_step,
                "trend": self.get_trend(name) if self.metrics[name] else None,
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {path}")

    def plot_metric(
        self,
        metric_name: str,
        ax: Optional[plt.Axes] = None,
        show_ma: bool = True,
        show_best: bool = True,
    ) -> plt.Axes:
        """
        Plot a single metric with optional moving average overlay and best marker.

        Args:
            metric_name: Name of the metric to plot
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            show_ma: Whether to overlay the moving average line
            show_best: Whether to mark the best value with a star

        Returns:
            The matplotlib Axes object used for plotting.

        Raises:
            KeyError: If metric_name has not been recorded.
        """
        if metric_name not in self.metrics:
            raise KeyError(f"No metric named '{metric_name}'")

        values = self.metrics[metric_name]
        steps = self.steps[metric_name]

        if not values:
            raise ValueError(f"No values recorded for '{metric_name}'")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        # Raw values as transparent line
        ax.plot(steps, values, "-", alpha=0.35, color="#1f77b4", linewidth=1.0, label="Raw")

        # Moving average as bold line
        if show_ma and len(values) > 1:
            ma = self.get_moving_average(metric_name)
            ax.plot(steps, ma, "-", color="#1f77b4", linewidth=2.0, label=f"MA({self.window_size})")

        # Best point marker
        if show_best:
            is_loss = "loss" in metric_name.lower()
            mode = "min" if is_loss else "max"
            best_val, best_step = self.get_best(metric_name, mode=mode)
            ax.plot(
                best_step,
                best_val,
                "*",
                color="#d62728",
                markersize=14,
                label=f"Best: {best_val:.4f}",
                zorder=5,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_all(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot all tracked metrics in a grid layout.

        Args:
            figsize: Figure size (width, height)
        """
        metric_names = [name for name in self.metrics if self.metrics[name]]
        n = len(metric_names)
        if n == 0:
            logger.warning("No metrics to plot.")
            return

        cols = min(3, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

        for idx, name in enumerate(metric_names):
            row, col = divmod(idx, cols)
            self.plot_metric(name, ax=axes[row][col])

        # Hide unused subplots
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        plt.show()
