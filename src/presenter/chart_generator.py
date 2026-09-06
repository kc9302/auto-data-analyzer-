"""
Chart Generator Module
Generates high-resolution, visually polished charts for embedding into PowerPoint (PPTX) and HTML reports.
Uses modern styling: Slate/Emerald/Sky palette, clean spines, data labels.
"""
import os
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class PresentationChartGenerator:
    def __init__(self, theme_colors: Dict[str, str] = None):
        self.colors = theme_colors or {
            "primary": "#1E293B",
            "accent": "#0EA5E9",
            "success": "#10B981",
            "warning": "#F59E0B",
            "danger": "#F43F5E",
            "bg": "#F8FAFC",
            "card_bg": "#FFFFFF",
            "muted": "#64748B"
        }
        # Configure clean matplotlib rcParams
        plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False

    def generate_missing_chart(self, missing_summary: List[Dict[str, Any]], out_path: str):
        if not missing_summary:
            # Create a clean "Zero Missing" badge chart
            fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
            ax.text(0.5, 0.5, "✓ 100% Complete Data\n(No Missing Values Detected)", 
                    ha="center", va="center", fontsize=14, fontweight="bold", color=self.colors["success"])
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, transparent=False, facecolor="#FFFFFF")
            plt.close()
            return

        cols = [m["column"] for m in missing_summary[:7]][::-1]
        ratios = [float(m["missing_ratio"]) for m in missing_summary[:7]][::-1]
        bar_colors = [self.colors["danger"] if r >= 40 else (self.colors["warning"] if r >= 5 else self.colors["accent"]) for r in ratios]

        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
        bars = ax.barh(cols, ratios, color=bar_colors, height=0.55, edgecolor="none", zorder=3)
        ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        ax.set_xlim(0, max(max(ratios) * 1.25, 10))
        ax.set_xlabel("Missing Ratio (%)", fontsize=10, color=self.colors["muted"], fontweight="bold")
        ax.set_title("Column-Level Missingness & Governance Tiers", fontsize=11, fontweight="bold", color=self.colors["primary"], pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")

        # Value labels
        for bar, r in zip(bars, ratios):
            ax.text(r + 0.5, bar.get_y() + bar.get_height() / 2, f"{r:.1f}%", 
                    va="center", fontsize=9, fontweight="bold", color=self.colors["primary"])

        plt.tight_layout()
        plt.savefig(out_path, transparent=False, facecolor="#FFFFFF")
        plt.close()

    def generate_ab_test_chart(self, ab_test_data: Dict[str, Any], out_path: str):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
        if not ab_test_data or "group_a" not in ab_test_data:
            ax.text(0.5, 0.5, "A/B Benchmark Pending", ha="center", va="center", color=self.colors["muted"])
            ax.axis("off")
            plt.savefig(out_path)
            plt.close()
            return

        folds = [f"F{i+1}" for i in range(len(ab_test_data["group_a"]["fold_scores"]))] + ["Avg"]
        scores_a = ab_test_data["group_a"]["fold_scores"] + [ab_test_data["group_a"]["mean_score"]]
        scores_b = ab_test_data["group_b"]["fold_scores"] + [ab_test_data["group_b"]["mean_score"]]

        x = np.arange(len(folds))
        width = 0.35

        bars_a = ax.bar(x - width/2, scores_a, width, label="Group A (Baseline)", color="#94A3B8", zorder=3)
        bars_b = ax.bar(x + width/2, scores_b, width, label="Group B (Engineered)", color=self.colors["success"], zorder=3)

        lift_val = ab_test_data.get("lift_pct", 0)
        metric = ab_test_data.get("metric_name", "Score")
        ax.set_ylabel(metric, fontsize=10, fontweight="bold", color=self.colors["muted"])
        ax.set_title(f"Feature A/B Test: +{lift_val}% Lift Across 5 CV Folds", fontsize=11, fontweight="bold", color=self.colors["primary"], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(folds, fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")

        # Annotation on average
        avg_a = scores_a[-1]
        avg_b = scores_b[-1]
        min_y = min(min(scores_a), min(scores_b)) * 0.95
        max_y = max(max(scores_a), max(scores_b)) * 1.08
        ax.set_ylim(min_y, max_y)

        plt.tight_layout()
        plt.savefig(out_path, transparent=False, facecolor="#FFFFFF")
        plt.close()

    def generate_leaderboard_chart(self, leaderboard: List[Dict[str, Any]], primary_metric: str, out_path: str):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
        if not leaderboard:
            ax.text(0.5, 0.5, "Leaderboard Empty", ha="center", va="center")
            ax.axis("off")
            plt.savefig(out_path)
            plt.close()
            return

        top_models = leaderboard[:5][::-1]
        names = [m["model"] for m in top_models]
        scores = [float(m.get(primary_metric, 0.0)) for m in top_models]
        colors = [self.colors["success"] if m.get("rank") == 1 else self.colors["accent"] for m in top_models]

        bars = ax.barh(names, scores, color=colors, height=0.55, zorder=3)
        ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        ax.set_title(f"Model Tournament Leaderboard ({primary_metric})", fontsize=11, fontweight="bold", color=self.colors["primary"], pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")

        for bar, s in zip(bars, scores):
            ax.text(s + (max(scores)*0.02 if max(scores) > 0 else 0.01), bar.get_y() + bar.get_height() / 2,
                    f"{s:.4f}", va="center", fontsize=9, fontweight="bold", color=self.colors["primary"])

        min_x = min(scores) * 0.9 if min(scores) > 0 else 0
        max_x = max(scores) * 1.2 if max(scores) > 0 else 1.0
        ax.set_xlim(min_x, max_x)

        plt.tight_layout()
        plt.savefig(out_path, transparent=False, facecolor="#FFFFFF")
        plt.close()

    def generate_importance_chart(self, top_features: List[Tuple[str, float]], out_path: str):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
        if not top_features:
            ax.text(0.5, 0.5, "No Feature Importance Data", ha="center", va="center")
            ax.axis("off")
            plt.savefig(out_path)
            plt.close()
            return

        features = [f[0] for f in top_features[:8]][::-1]
        vals = [float(f[1]) for f in top_features[:8]][::-1]

        # Highlight synthetic features in Sky Blue, raw features in Slate Navy
        colors = [self.colors["accent"] if ("ratio_" in f or "log1p_" in f or "_dev_" in f or "_is_missing" in f) else self.colors["primary"] for f in features]

        bars = ax.barh(features, vals, color=colors, height=0.55, zorder=3)
        ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        ax.set_xlabel("Relative Importance Weight", fontsize=10, color=self.colors["muted"], fontweight="bold")
        ax.set_title("Top 8 Influential Features (Cyan = Engineered)", fontsize=11, fontweight="bold", color=self.colors["primary"], pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")

        for bar, v in zip(bars, vals):
            ax.text(v + (max(vals)*0.02), bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=9, fontweight="bold", color=self.colors["primary"])

        ax.set_xlim(0, max(vals) * 1.25)
        plt.tight_layout()
        plt.savefig(out_path, transparent=False, facecolor="#FFFFFF")
        plt.close()
