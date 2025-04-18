#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for Hybrid Bug Predictor

Provides unified visualization capabilities for FixCache, REPD, and hybrid
bug prediction results.

Author: anirudhsengar
"""

import os
import logging
import math
import json
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logger.warning(f"Visualization libraries not available: {str(e)}. "
                   "Some visualization functions will be limited.")


class Visualizer:
    """
    Unified visualizer for bug prediction results.

    This class provides visualization methods for FixCache, REPD,
    and hybrid prediction approaches.
    """

    def __init__(self,
                 output_dir: str = "results",
                 show_plots: bool = False,
                 style: str = "darkgrid",
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
            show_plots: Whether to display plots (in addition to saving them)
            style: Seaborn style for plots
            figsize: Default figure size
        """
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.figsize = figsize

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up visualization environment if available
        if VISUALIZATION_AVAILABLE:
            # Set plot style
            sns.set_style(style)
            # Use a color palette that works well for all types of vision
            self.colors = {
                "fixcache": "#1f77b4",  # Blue
                "repd": "#ff7f0e",  # Orange
                "hybrid": "#2ca02c",  # Green
                "overlap": "#9467bd",  # Purple
                "background": "#d3d3d3"  # Light gray
            }

    def _check_visualization_available(self) -> bool:
        """
        Check if visualization libraries are available.

        Returns:
            True if visualization libraries are available, False otherwise
        """
        if not VISUALIZATION_AVAILABLE:
            logger.error("Visualization libraries (matplotlib, seaborn, etc.) not available.")
            return False
        return True

    # ===== FixCache Visualization Methods =====

    def visualize_fixcache_results(self,
                                   cached_files: List[str],
                                   all_files: List[str],
                                   bug_fixes: List[str],
                                   output_file: str = "fixcache_results.png",
                                   title: str = "FixCache Results") -> bool:
        """
        Visualize FixCache results.

        Args:
            cached_files: List of files in the cache
            all_files: List of all files in the repository
            bug_fixes: List of files involved in bug fixes
            output_file: Output file path (relative to output_dir)
            title: Plot title

        Returns:
            True if successful, False otherwise
        """
        if not self._check_visualization_available():
            return False

        try:
            plt.figure(figsize=self.figsize)

            # Calculate metrics
            total_files = len(all_files)
            cached_file_count = len(cached_files)
            bug_fix_count = len(bug_fixes)

            # Calculate overlap between cache and actual bug fixes
            cached_bug_fixes = set(cached_files).intersection(set(bug_fixes))
            hit_rate = len(cached_bug_fixes) / bug_fix_count if bug_fix_count > 0 else 0

            # Create a Venn-like diagram showing cache, bug fixes, and overlap
            plt.figure(figsize=self.figsize)

            # Create a 2x2 grid
            ax = plt.subplot(111)

            # Draw total files box (background)
            ax.add_patch(plt.Rectangle((0, 0), 100, 100,
                                       fill=True, color=self.colors["background"],
                                       alpha=0.3, label=f"All Files: {total_files}"))

            # Draw bug fixes box
            ax.add_patch(plt.Rectangle((10, 10), 40, 80,
                                       fill=True, color=self.colors["fixcache"],
                                       alpha=0.5, label=f"Bug Fixes: {bug_fix_count}"))

            # Draw cache box
            ax.add_patch(plt.Rectangle((30, 20), 60, 60,
                                       fill=True, color=self.colors["repd"],
                                       alpha=0.5, label=f"Cache: {cached_file_count}"))

            # Overlap text
            plt.text(50, 50, f"Hit Rate:\n{hit_rate:.2%}",
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14, fontweight='bold')

            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.legend(loc='upper right')

            # Add metrics text
            metrics_text = (
                f"Cache Size: {cached_file_count} files ({cached_file_count / total_files:.1%} of total)\n"
                f"Bug Fixes: {bug_fix_count} files\n"
                f"Hits: {len(cached_bug_fixes)} files\n"
                f"Hit Rate: {hit_rate:.2%}"
            )
            plt.figtext(0.02, 0.02, metrics_text, fontsize=12)

            # Save the figure
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved FixCache visualization to {output_path}")

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error creating FixCache visualization: {str(e)}")
            return False

    # ===== REPD Visualization Methods =====

    def visualize_repd_results(self,
                               risk_scores: Dict[str, float],
                               bug_fixes: List[str],
                               output_file: str = "repd_results.png",
                               title: str = "REPD Results") -> bool:
        """
        Visualize REPD risk scores.

        Args:
            risk_scores: Dictionary mapping file paths to risk scores
            bug_fixes: List of files involved in bug fixes
            output_file: Output file path (relative to output_dir)
            title: Plot title

        Returns:
            True if successful, False otherwise
        """
        if not self._check_visualization_available():
            return False

        try:
            # Create a DataFrame for easier analysis
            df = pd.DataFrame({
                'file': list(risk_scores.keys()),
                'risk_score': list(risk_scores.values()),
                'is_bug_fix': [file in bug_fixes for file in risk_scores.keys()]
            })

            # Create the visualization with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

            # Plot 1: Distribution of risk scores
            sns.histplot(data=df, x='risk_score', hue='is_bug_fix',
                         multiple='stack', kde=True, ax=ax1)
            ax1.set_title('Distribution of Risk Scores')
            ax1.set_xlabel('Risk Score')
            ax1.set_ylabel('Frequency')
            ax1.legend(['Non-Bug Files', 'Bug Files'])

            # Plot 2: ROC-like curve (sort of)
            # Sort files by risk score
            sorted_df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)
            sorted_df['cumulative_bugs'] = sorted_df['is_bug_fix'].cumsum() / sum(sorted_df['is_bug_fix'])
            sorted_df['files_checked'] = (sorted_df.index + 1) / len(sorted_df)

            ax2.plot(sorted_df['files_checked'], sorted_df['cumulative_bugs'])
            ax2.plot([0, 1], [0, 1], 'k--')  # Random guess line
            ax2.set_title('Bug Detection Rate')
            ax2.set_xlabel('Proportion of Files Checked')
            ax2.set_ylabel('Proportion of Bugs Found')

            # Add AUC-like metric
            auc = np.trapz(sorted_df['cumulative_bugs'], sorted_df['files_checked'])
            ax2.text(0.6, 0.3, f'AUC: {auc:.3f}', fontsize=12)

            # Add overall title
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved REPD visualization to {output_path}")

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error creating REPD visualization: {str(e)}")
            return False

    def visualize_reconstruction_errors(self,
                                        reconstruction_errors: Dict[str, float],
                                        highlight_files: Optional[List[str]] = None,
                                        output_file: str = "reconstruction_errors.png",
                                        title: str = "Reconstruction Errors") -> bool:
        """
        Visualize reconstruction errors from REPD.

        Args:
            reconstruction_errors: Dictionary mapping file paths to reconstruction errors
            highlight_files: List of files to highlight (e.g., bug fixes)
            output_file: Output file path (relative to output_dir)
            title: Plot title

        Returns:
            True if successful, False otherwise
        """
        if not self._check_visualization_available():
            return False

        try:
            # Create data for plotting
            errors = list(reconstruction_errors.values())
            files = list(reconstruction_errors.keys())

            # Create highlighting data
            highlight = [file in (highlight_files or []) for file in files]

            # Create DataFrame
            df = pd.DataFrame({
                'file': files,
                'error': errors,
                'highlight': highlight
            })

            # Sort by error
            df = df.sort_values('error', ascending=False)

            # Limit to top 100 for visibility
            if len(df) > 100:
                df = df.head(100)

            # Plot
            plt.figure(figsize=self.figsize)
            bars = plt.bar(range(len(df)), df['error'],
                           color=[self.colors['repd'] if h else 'gray' for h in df['highlight']])

            # Add labels and title
            plt.title(title, fontsize=16)
            plt.xlabel('Files (sorted by reconstruction error)', fontsize=12)
            plt.ylabel('Reconstruction Error', fontsize=12)
            plt.xticks([])  # Hide x-axis ticks

            # Add legend
            if highlight_files:
                plt.legend([
                    Patch(facecolor='gray', label='Regular files'),
                    Patch(facecolor=self.colors['repd'], label='Highlighted files')
                ])

            # Save the figure
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved reconstruction errors visualization to {output_path}")

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error creating reconstruction errors visualization: {str(e)}")
            return False

    # ===== Hybrid Visualization Methods =====

    def visualize_hybrid_results(self,
                                 hybrid_scores: Dict[str, float],
                                 fixcache_files: Set[str],
                                 repd_scores: Dict[str, float],
                                 bug_fixes: List[str],
                                 weights: Dict[str, float],
                                 output_file: str = "hybrid_results.png",
                                 title: str = "Hybrid Prediction Results") -> bool:
        """
        Visualize hybrid prediction results.

        Args:
            hybrid_scores: Dictionary mapping file paths to hybrid scores
            fixcache_files: Set of files in the FixCache
            repd_scores: Dictionary mapping file paths to REPD risk scores
            bug_fixes: List of files involved in bug fixes
            weights: Dictionary with weights used for each approach
            output_file: Output file path (relative to output_dir)
            title: Plot title

        Returns:
            True if successful, False otherwise
        """
        if not self._check_visualization_available():
            return False

        try:
            # Create a DataFrame with all results
            all_files = set(hybrid_scores.keys()) | fixcache_files | set(repd_scores.keys())

            data = []
            for file in all_files:
                data.append({
                    'file': file,
                    'fixcache': 1.0 if file in fixcache_files else 0.0,
                    'repd': repd_scores.get(file, 0.0),
                    'hybrid': hybrid_scores.get(file, 0.0),
                    'is_bug_fix': file in bug_fixes
                })

            df = pd.DataFrame(data)

            # Create a figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Hybrid scores distribution
            sns.histplot(data=df, x='hybrid', hue='is_bug_fix',
                         multiple='stack', kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Hybrid Score Distribution')
            axes[0, 0].set_xlabel('Hybrid Score')
            axes[0, 0].set_ylabel('Frequency')

            # Plot 2: Comparison of scores for bug fixes
            bug_df = df[df['is_bug_fix']].melt(
                id_vars=['file', 'is_bug_fix'],
                value_vars=['fixcache', 'repd', 'hybrid'],
                var_name='approach', value_name='score'
            )

            sns.boxplot(data=bug_df, x='approach', y='score', ax=axes[0, 1])
            axes[0, 1].set_title('Score Distribution for Bug Files')
            axes[0, 1].set_xlabel('Approach')
            axes[0, 1].set_ylabel('Score')

            # Plot 3: Bug detection effectiveness
            approaches = ['fixcache', 'repd', 'hybrid']
            colors = [self.colors['fixcache'], self.colors['repd'], self.colors['hybrid']]

            for approach, color in zip(approaches, colors):
                # Sort by the approach's score
                sorted_df = df.sort_values(approach, ascending=False).reset_index(drop=True)
                sorted_df['cumulative_bugs'] = sorted_df['is_bug_fix'].cumsum() / sum(sorted_df['is_bug_fix'])
                sorted_df['files_checked'] = (sorted_df.index + 1) / len(sorted_df)

                axes[1, 0].plot(sorted_df['files_checked'], sorted_df['cumulative_bugs'],
                                label=approach.capitalize(), color=color)

            # Add random guess line
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1, 0].set_title('Approach Comparison')
            axes[1, 0].set_xlabel('Proportion of Files Checked')
            axes[1, 0].set_ylabel('Proportion of Bugs Found')
            axes[1, 0].legend()

            # Plot 4: Weights visualization
            weight_df = pd.DataFrame({
                'Approach': list(weights.keys()),
                'Weight': list(weights.values())
            })

            sns.barplot(data=weight_df, x='Approach', y='Weight', ax=axes[1, 1],
                        palette=[self.colors['fixcache'], self.colors['repd']])
            axes[1, 1].set_title('Hybrid Weights')
            axes[1, 1].set_xlabel('Approach')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].set_ylim(0, 1.0)

            # Add weights as text on bars
            for i, weight in enumerate(weights.values()):
                axes[1, 1].text(i, weight + 0.05, f'{weight:.2f}',
                                ha='center', va='bottom')

            # Add AUC values to the approach comparison plot
            for approach, color in zip(approaches, colors):
                sorted_df = df.sort_values(approach, ascending=False).reset_index(drop=True)
                sorted_df['cumulative_bugs'] = sorted_df['is_bug_fix'].cumsum() / sum(sorted_df['is_bug_fix'])
                sorted_df['files_checked'] = (sorted_df.index + 1) / len(sorted_df)

                auc = np.trapz(sorted_df['cumulative_bugs'], sorted_df['files_checked'])
                axes[1, 0].text(0.7, 0.3 - 0.05 * approaches.index(approach),
                                f'{approach.capitalize()}: {auc:.3f}',
                                fontsize=10, color=color)

            # Add overall title
            plt.suptitle(title, fontsize=18)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            # Save the figure
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved hybrid visualization to {output_path}")

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error creating hybrid visualization: {str(e)}")
            return False

    def visualize_top_files(self,
                            top_files: List[Tuple[str, float]],
                            bug_fixes: Optional[List[str]] = None,
                            approach: str = "hybrid",
                            output_file: str = "top_risky_files.png",
                            title: str = "Top Risky Files") -> bool:
        """
        Visualize top risky files from any approach.

        Args:
            top_files: List of (file_path, score) tuples
            bug_fixes: Optional list of known bug files to highlight
            approach: Approach name ('fixcache', 'repd', or 'hybrid')
            output_file: Output file path (relative to output_dir)
            title: Plot title

        Returns:
            True if successful, False otherwise
        """
        if not self._check_visualization_available():
            return False

        try:
            # Choose color based on approach
            if approach.lower() == "fixcache":
                color = self.colors["fixcache"]
            elif approach.lower() == "repd":
                color = self.colors["repd"]
            else:
                color = self.colors["hybrid"]

            # Create data for plotting
            files = [f[0] for f in top_files]
            scores = [f[1] for f in top_files]

            # Determine if files are known bugs
            is_bug = [file in (bug_fixes or []) for file in files]

            # Create short file names for plotting
            short_names = [self._shorten_filename(file) for file in files]

            # Create the plot
            plt.figure(figsize=(10, len(top_files) * 0.4 + 2))

            # Create horizontal bar chart
            bars = plt.barh(range(len(top_files)), scores,
                            color=[self.colors["fixcache"] if bug else color for bug in is_bug])

            # Add file names
            for i, (name, score) in enumerate(zip(short_names, scores)):
                plt.text(0.02, i, name, ha='left', va='center', color='white')
                plt.text(score + 0.02, i, f"{score:.3f}", ha='left', va='center')

            # Set up axes
            plt.yticks([])
            plt.title(title, fontsize=16)
            plt.xlabel('Risk Score', fontsize=12)

            # Add legend
            if bug_fixes:
                plt.legend([
                    Patch(facecolor=color, label='Predicted Risky Files'),
                    Patch(facecolor=self.colors["fixcache"], label='Known Bug Files')
                ])

            # Save the figure
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved top files visualization to {output_path}")

            if self.show_plots:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            logger.error(f"Error creating top files visualization: {str(e)}")
            return False

    # ===== Combined Report Generation =====

    def generate_comparison_report(self,
                                   hybrid_scores: Dict[str, float],
                                   fixcache_files: Set[str],
                                   repd_scores: Dict[str, float],
                                   bug_fixes: List[str],
                                   weights: Dict[str, float],
                                   output_file: str = "comparison_report") -> bool:
        """
        Generate a comprehensive comparison report of all approaches.

        Args:
            hybrid_scores: Dictionary mapping file paths to hybrid scores
            fixcache_files: Set of files in the FixCache
            repd_scores: Dictionary mapping file paths to REPD risk scores
            bug_fixes: List of files involved in bug fixes
            weights: Dictionary with weights used for each approach
            output_file: Base output filename (without extension)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a DataFrame with all results
            all_files = set(hybrid_scores.keys()) | fixcache_files | set(repd_scores.keys())

            data = []
            for file in all_files:
                # Skip files that don't exist in any approach
                if not (file in hybrid_scores or file in fixcache_files or file in repd_scores):
                    continue

                data.append({
                    'file': file,
                    'fixcache_score': 1.0 if file in fixcache_files else 0.0,
                    'repd_score': repd_scores.get(file, 0.0),
                    'hybrid_score': hybrid_scores.get(file, 0.0),
                    'is_bug_fix': file in bug_fixes
                })

            df = pd.DataFrame(data)

            # Calculate performance metrics for each approach
            metrics = self._calculate_performance_metrics(df, bug_fixes)

            # Generate HTML report if visualization libraries are available
            if VISUALIZATION_AVAILABLE:
                self._generate_html_report(df, metrics, weights, output_file)

            # Always generate CSV report
            self._generate_csv_report(df, metrics, output_file)

            # Generate JSON metrics
            self._generate_json_metrics(metrics, weights, output_file)

            return True

        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return False

    def _calculate_performance_metrics(self,
                                       df: pd.DataFrame,
                                       bug_fixes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each approach.

        Args:
            df: DataFrame containing prediction scores
            bug_fixes: List of known bug files

        Returns:
            Dictionary of metrics for each approach
        """
        approaches = ['fixcache_score', 'repd_score', 'hybrid_score']
        metrics = {}

        for approach in approaches:
            approach_name = approach.split('_')[0]
            metrics[approach_name] = {}

            # Sort files by score for this approach
            sorted_df = df.sort_values(approach, ascending=False).reset_index(drop=True)

            # Calculate cumulative bugs found
            sorted_df['cumulative_bugs'] = sorted_df['is_bug_fix'].cumsum()
            total_bugs = sorted_df['is_bug_fix'].sum()

            if total_bugs > 0:
                # AUC for bug detection
                sorted_df['norm_bugs'] = sorted_df['cumulative_bugs'] / total_bugs
                sorted_df['norm_files'] = (sorted_df.index + 1) / len(sorted_df)
                auc = np.trapz(sorted_df['norm_bugs'], sorted_df['norm_files'])
                metrics[approach_name]['auc'] = auc

                # Calculate metrics at different thresholds
                thresholds = [0.1, 0.2, 0.5]  # Examine top 10%, 20%, 50% of files

                for threshold in thresholds:
                    n_files = int(len(sorted_df) * threshold)
                    top_files = sorted_df.head(n_files)

                    # Calculate recall at this threshold
                    bugs_found = top_files['is_bug_fix'].sum()
                    recall = bugs_found / total_bugs

                    metrics[approach_name][f'recall@{int(threshold * 100)}%'] = recall

        return metrics

    def _generate_html_report(self,
                              df: pd.DataFrame,
                              metrics: Dict[str, Dict[str, float]],
                              weights: Dict[str, float],
                              output_file: str) -> None:
        """
        Generate HTML report with interactive visualizations.

        Args:
            df: DataFrame containing prediction scores
            metrics: Dictionary of metrics for each approach
            weights: Dictionary with weights used for each approach
            output_file: Base output filename (without extension)
        """
        import base64
        from io import BytesIO

        # Create a styled HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bug Prediction Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #4CAF50; color: white; padding: 10px; }
                .section { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { text-align: left; padding: 8px; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart { margin: 20px 0; }
                .footer { font-size: 0.8em; color: #666; margin-top: 30px; }
            </style>
        </head>
        <body>
        """

        # Add header
        html += f"""
        <div class="header">
            <h1>Bug Prediction Comparison Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

        # Add summary metrics section
        html += """
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>FixCache</th>
                    <th>REPD</th>
                    <th>Hybrid</th>
                </tr>
        """

        # Add rows for each metric
        all_metrics = set()
        for approach_metrics in metrics.values():
            all_metrics.update(approach_metrics.keys())

        for metric in sorted(all_metrics):
            html += f"<tr><td>{metric}</td>"
            for approach in ['fixcache', 'repd', 'hybrid']:
                value = metrics.get(approach, {}).get(metric, "N/A")
                if isinstance(value, float):
                    html += f"<td>{value:.3f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"

        html += """
            </table>
        </div>
        """

        # Add weights section
        html += """
        <div class="section">
            <h2>Hybrid Weights</h2>
            <table>
                <tr>
                    <th>Approach</th>
                    <th>Weight</th>
                </tr>
        """

        for approach, weight in weights.items():
            html += f"<tr><td>{approach}</td><td>{weight:.3f}</td></tr>"

        html += """
            </table>
        </div>
        """

        # Add visualizations
        html += """
        <div class="section">
            <h2>Visualizations</h2>
        """

        # Create and embed bug detection curve
        if VISUALIZATION_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))

            for approach, color in zip(['fixcache_score', 'repd_score', 'hybrid_score'],
                                       [self.colors['fixcache'], self.colors['repd'], self.colors['hybrid']]):
                approach_name = approach.split('_')[0]

                # Sort by the approach's score
                sorted_df = df.sort_values(approach, ascending=False).reset_index(drop=True)
                sorted_df['cumulative_bugs'] = sorted_df['is_bug_fix'].cumsum() / sum(sorted_df['is_bug_fix'])
                sorted_df['files_checked'] = (sorted_df.index + 1) / len(sorted_df)

                ax.plot(sorted_df['files_checked'], sorted_df['cumulative_bugs'],
                        label=approach_name.capitalize(), color=color)

                # Add AUC value
                auc = metrics.get(approach_name, {}).get('auc', 0)
                ax.text(0.7, 0.3 - 0.05 * ['fixcache_score', 'repd_score', 'hybrid_score'].index(approach),
                        f'{approach_name.capitalize()}: {auc:.3f}',
                        fontsize=10, color=color)

            # Add random guess line
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_title('Approach Comparison')
            ax.set_xlabel('Proportion of Files Checked')
            ax.set_ylabel('Proportion of Bugs Found')
            ax.legend()

            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()

            # Encode to base64
            data = base64.b64encode(buf.getbuffer()).decode('ascii')

            # Add to HTML
            html += f"""
            <div class="chart">
                <h3>Bug Detection Effectiveness</h3>
                <img src="data:image/png;base64,{data}" alt="Bug Detection Curve">
            </div>
            """

        html += """
        </div>
        """

        # Add top files from each approach
        html += """
        <div class="section">
            <h2>Top 10 Risky Files</h2>
        """

        for approach in ['fixcache_score', 'repd_score', 'hybrid_score']:
            approach_name = approach.split('_')[0]

            html += f"""
            <h3>Top Files by {approach_name.capitalize()}</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>File</th>
                    <th>Score</th>
                    <th>Is Bug Fix</th>
                </tr>
            """

            # Sort by the approach's score
            top_files = df.sort_values(approach, ascending=False).head(10)

            for i, (_, row) in enumerate(top_files.iterrows(), 1):
                html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['file']}</td>
                    <td>{row[approach]:.3f}</td>
                    <td>{'Yes' if row['is_bug_fix'] else 'No'}</td>
                </tr>
                """

            html += """
            </table>
            """

        html += """
        </div>
        """

        # Add footer
        html += """
        <div class="footer">
            <p>Generated by GlitchWitcher Hybrid Bug Predictor</p>
            <p>Author: anirudhsengar</p>
        </div>
        </body>
        </html>
        """

        # Write HTML to file
        output_path = os.path.join(self.output_dir, f"{output_file}.html")
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Saved HTML report to {output_path}")

    def _generate_csv_report(self,
                             df: pd.DataFrame,
                             metrics: Dict[str, Dict[str, float]],
                             output_file: str) -> None:
        """
        Generate CSV report with all data.

        Args:
            df: DataFrame containing prediction scores
            metrics: Dictionary of metrics for each approach
            output_file: Base output filename (without extension)
        """
        # Save the full DataFrame
        output_path = os.path.join(self.output_dir, f"{output_file}_full.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved full CSV report to {output_path}")

        # Create and save metrics DataFrame
        metrics_data = []

        for approach, approach_metrics in metrics.items():
            for metric, value in approach_metrics.items():
                metrics_data.append({
                    'approach': approach,
                    'metric': metric,
                    'value': value
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(self.output_dir, f"{output_file}_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics CSV report to {metrics_path}")

    def _generate_json_metrics(self,
                               metrics: Dict[str, Dict[str, float]],
                               weights: Dict[str, float],
                               output_file: str) -> None:
        """
        Generate JSON metrics file.

        Args:
            metrics: Dictionary of metrics for each approach
            weights: Dictionary with weights used for each approach
            output_file: Base output filename (without extension)
        """
        result = {
            'metrics': metrics,
            'weights': weights,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        output_path = os.path.join(self.output_dir, f"{output_file}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved JSON metrics to {output_path}")

    # ===== Helper Methods =====

    def _shorten_filename(self, filename: str, max_length: int = 40) -> str:
        """
        Shorten a filename for display purposes.

        Args:
            filename: Original filename
            max_length: Maximum length of shortened filename

        Returns:
            Shortened filename
        """
        if len(filename) <= max_length:
            return filename

        # Split into directory and filename
        parts = filename.split('/')

        if len(parts) <= 2:
            # Just filename or parent directory + filename
            return "..." + filename[-(max_length - 3):]

        # Keep first directory, last directory, and filename
        return f"{parts[0]}/.../{parts[-2]}/{parts[-1]}"


# Simple function to create a visualizer with default settings
def create_visualizer(output_dir: str = "results", show_plots: bool = False) -> Visualizer:
    """
    Create a visualizer with default settings.

    Args:
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots (in addition to saving them)

    Returns:
        Visualizer object
    """
    return Visualizer(output_dir=output_dir, show_plots=show_plots)