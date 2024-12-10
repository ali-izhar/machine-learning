#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.interpolate import BSpline


@dataclass
class PolynomialPiece:
    """Represents a piece of a piecewise polynomial function."""

    x_pos: float
    equation: str
    interval: Tuple[float, float]
    eval_func: callable


class SplineVisualizer:
    """Class for visualizing different types of splines."""

    def __init__(self):
        """Initialize visualization settings."""
        self.style = "whitegrid"
        self.figsize = (12, 8)
        self.colors = ["#FF9933", "#33CC33", "#9933FF"]
        self.knot_color = "#E74C3C"
        self.annotation_colors = {
            "equation": "#F7DC6F",
            "knot": "#82E0AA",
            "arrow": "#7D3C98",
        }

    def _setup_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plotting environment."""
        sns.set_style(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        return fig, ax

    def _add_polynomial_annotation(
        self, ax: plt.Axes, piece: PolynomialPiece, y_pos: float
    ) -> None:
        """Add annotation for a polynomial piece."""
        ax.annotate(
            piece.equation,
            xy=(piece.x_pos, y_pos),
            xytext=(piece.x_pos, y_pos + 0.5),
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc=self.annotation_colors["equation"],
                alpha=0.7,
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0",
                color=self.annotation_colors["arrow"],
            ),
        )

    def _add_knot_annotation(self, ax: plt.Axes, x: float, y: float, idx: int) -> None:
        """Add annotation for a knot point."""
        ax.annotate(
            f"x_{idx}",
            xy=(x, y),
            xytext=(x - 0.2, y - 0.3),
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3", fc=self.annotation_colors["knot"], alpha=0.7
            ),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#229954"
            ),
        )

    def draw_piecewise_spline(self, save_path: Optional[str] = None) -> plt.Figure:
        """Draw a piecewise spline with smooth polynomial connections."""
        # Define knot points with their exact y-values
        knots = [
            (0, 0),  # y₁(0) = 0
            (1, 1),  # y₁(1) = y₂(1) = 1
            (2, 0.5),  # y₂(2) = -0.5(1)² + 1 = 0.5
            (3, 0.7),  # y₃(3) = 0.2(27) - 1.2(9) + 2.4(3) - 1.1 = 0.7
        ]
        x = np.array([k[0] for k in knots])
        y = np.array([k[1] for k in knots])

        # Define polynomial pieces with the exact functions
        pieces = [
            PolynomialPiece(0.5, "-x² + 2x", (0, 1), lambda x: -(x**2) + 2 * x),
            PolynomialPiece(
                1.5, "-0.5(x-1)² + 1", (1, 2), lambda x: -0.5 * (x - 1) ** 2 + 1
            ),
            PolynomialPiece(
                2.5,
                "0.2x³ - 1.2x² + 2.4x - 1.1",
                (2, 3),
                lambda x: 0.2 * x**3 - 1.2 * x**2 + 2.4 * x - 1.1,
            ),
        ]

        # Setup plot
        fig, ax = self._setup_plot()

        # Plot knot points
        ax.plot(x, y, "o", markersize=10, label="Knots", color=self.knot_color)

        # Plot polynomial segments
        for piece, color in zip(pieces, self.colors):
            x_seg = np.linspace(*piece.interval, 50)
            y_seg = piece.eval_func(x_seg)

            # Plot segment
            ax.plot(x_seg, y_seg, "-", linewidth=2.5, color=color, alpha=0.8)

            # Add annotation
            y_pos = piece.eval_func(piece.x_pos)
            self._add_polynomial_annotation(ax, piece, y_pos)

        # Add knot annotations with exact positions
        for i, (x_val, y_val) in enumerate(knots):
            self._add_knot_annotation(ax, x_val, y_val, i)

        # Customize plot
        ax.set_title("Piecewise Polynomial Functions", fontsize=14, pad=20)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(loc="upper right")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(-0.5, 3.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def draw_bspline(self, save_path: Optional[str] = None) -> plt.Figure:
        """Draw a B-spline with its basis functions."""
        # Setup plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Define knot vector (clamped B-spline)
        degree = 3
        knots = np.array([0] * degree + [0, 1, 2, 3] + [3] * degree)

        # Define control points (x and y coordinates separately)
        x_points = np.array([0, 0, 1, 2, 3, 3])
        y_points = np.array([0, 1, 2, 2, 1, 0])

        # Create B-spline objects for x and y coordinates
        bspline_x = BSpline(knots, x_points, degree)
        bspline_y = BSpline(knots, y_points, degree)

        # Plot the B-spline curve
        t = np.linspace(0, 3, 100)
        x_curve = bspline_x(t)
        y_curve = bspline_y(t)

        # Upper plot: B-spline curve
        ax1.plot(
            x_curve, y_curve, "-", label="B-spline curve", color="#2E86C1", linewidth=2
        )
        ax1.plot(x_points, y_points, "o--", label="Control points", color="#E74C3C")
        ax1.set_title("Cubic B-spline Curve", fontsize=14, pad=20)
        ax1.grid(True)
        ax1.legend()

        # Lower plot: Basis functions
        colors = plt.cm.viridis(np.linspace(0, 1, len(x_points)))

        # Add vertical lines for knot positions
        unique_knots = np.unique(knots)
        for k in unique_knots:
            ax2.axvline(x=k, color="gray", linestyle="--", alpha=0.3)

        # Plot basis functions
        for i in range(len(x_points)):
            # Create basis function for each control point
            basis_coef = np.zeros(len(x_points))
            basis_coef[i] = 1.0
            basis = BSpline(knots, basis_coef, degree)

            # Plot basis function
            t_basis = np.linspace(0, 3, 200)
            ax2.plot(
                t_basis,
                basis(t_basis),
                "-",
                color=colors[i],
                label=f"B_{i}³(t)",
                alpha=0.7,
            )

        # Add knot annotations
        knot_multiplicities = {k: np.sum(knots == k) for k in unique_knots}
        y_pos = -0.1  # Position for knot labels
        for k in unique_knots:
            if knot_multiplicities[k] > 1:
                label = f"t={k}\n(×{knot_multiplicities[k]})"
            else:
                label = f"t={k}"
            ax2.text(k, y_pos, label, ha="center", va="top")

        ax2.set_title("B-spline Basis Functions", fontsize=14, pad=20)
        ax2.grid(True)
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel("t", fontsize=12)
        ax2.set_ylabel("Basis value", fontsize=12)
        ax2.set_ylim(-0.2, 1.1)  # Adjust y-limits to show knot labels

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualization tools for KAN concepts")
    parser.add_argument(
        "-spline", action="store_true", help="Draw a piecewise spline visualization"
    )
    parser.add_argument(
        "-bspline", action="store_true", help="Draw a B-spline visualization"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to save the visualization"
    )

    args = parser.parse_args()

    visualizer = SplineVisualizer()
    if args.spline:
        fig = visualizer.draw_piecewise_spline(args.output)
        plt.show()
    elif args.bspline:
        fig = visualizer.draw_bspline(args.output)
        plt.show()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
