"""Result visualization utilities."""
from typing import Any, Dict, List, Optional


class ResultVisualizer:
    """Visualize experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def plot_calibration_curve(self, predictions: List[float], actuals: List[bool],
                                save_path: str = None):
        """Plot reliability diagram."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_accs, bin_confs, bin_counts = [], [], []

            for i in range(n_bins):
                mask = [(bins[i] <= p < bins[i+1]) for p in predictions]
                if any(mask):
                    bin_preds = [p for p, m in zip(predictions, mask) if m]
                    bin_actuals = [float(a) for a, m in zip(actuals, mask) if m]
                    bin_accs.append(np.mean(bin_actuals))
                    bin_confs.append(np.mean(bin_preds))
                    bin_counts.append(len(bin_preds))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
            ax1.plot(bin_confs, bin_accs, 'o-', color='#2196F3', label='Agent')
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Calibration Curve')
            ax1.legend()

            ax2.bar(range(len(bin_counts)), bin_counts, color='#4CAF50', alpha=0.7)
            ax2.set_xlabel('Confidence Bin')
            ax2.set_ylabel('Count')
            ax2.set_title('Confidence Distribution')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except ImportError:
            pass

    def plot_metrics_over_time(self, metrics_history: List[Dict], save_path: str = None):
        """Plot metrics evolution over experiments."""
        try:
            import matplotlib.pyplot as plt

            steps = range(len(metrics_history))
            self_eval = [m.get("self_eval_accuracy", 0) for m in metrics_history]
            ece = [m.get("ece", 1) for m in metrics_history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(steps, self_eval, '-o', color='#2196F3')
            ax1.set_ylabel('Self-Eval Accuracy')
            ax1.set_title('Metacognition Quality Over Time')

            ax2.plot(steps, ece, '-o', color='#F44336')
            ax2.set_ylabel('ECE (lower=better)')
            ax2.set_xlabel('Experiment')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except ImportError:
            pass
