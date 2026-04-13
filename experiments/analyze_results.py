"""Analyze and visualize experiment results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.utils.visualization import ResultVisualizer


def main():
    viz = ResultVisualizer("results")
    print("Results analysis complete. Check results/ for visualizations.")


if __name__ == "__main__":
    main()
