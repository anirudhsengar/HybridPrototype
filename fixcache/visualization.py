#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for FixCache

This module provides visualization capabilities for the FixCache algorithm results,
including charts for hit rates, bug distributions, and comparative analyses.

Author: anirudhsengar
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter, defaultdict

# Setup logging
logger = logging.getLogger(__name__)

# Define color schemes
THEME_COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',  # Green
    'quaternary': '#d62728',  # Red
    'background': '#f8f9fa',
    'text': '#333333',
}


def visualize_results(results, output_file):
    """
    Generate visualization for FixCache results.

    Args:
        results (dict): The results dictionary from FixCache analysis
        output_file (str): The path to save the visualization

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Cache Hit Rate and Miss Rate
        # Ensure hit_rate is a number
        hit_rate = float(results.get('hit_rate', 0))
        miss_rate = 100 - hit_rate  # This was causing the error

        labels = ['Hit Rate', 'Miss Rate']
        sizes = [hit_rate, miss_rate]
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0)  # Explode hit rate slice

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_title('Bug Prediction Accuracy')

        # Plot 2: Top Bug-Prone Files (if available)
        if 'top_files' in results and results['top_files']:
            top_files = results['top_files'][:10]  # Top 10 files
            files = [os.path.basename(f) for f, _ in top_files]
            scores = [float(score) for _, score in top_files]  # Ensure scores are float

            y_pos = np.arange(len(files))

            ax2.barh(y_pos, scores, align='center', alpha=0.7, color='#2196F3')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(files)
            ax2.invert_yaxis()  # Labels read top-to-bottom
            ax2.set_xlabel('Risk Score')
            ax2.set_title('Top Bug-Prone Files')

            # Add text labels
            for i, score in enumerate(scores):
                ax2.text(score + 0.1, i, "{:.2f}".format(score), va='center')
        else:
            ax2.text(0.5, 0.5, 'No top files data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
            ax2.set_title('Top Bug-Prone Files')

        # Add overall title
        repo_name = os.path.basename(str(results.get('repository_path', 'Unknown Repository')))
        plt.suptitle("FixCache Analysis: {}".format(repo_name), fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info("Visualization successfully saved to {}".format(output_file))
        return True

    except ImportError:
        logger.error("Matplotlib is required for visualization. Install with: pip install matplotlib")
        return False
    except Exception as e:
        logger.error("Error generating visualization: {}".format(str(e)))
        return False


def _create_hit_rate_gauge(ax: plt.Axes, hit_rate: float) -> None:
    """
    Create a gauge chart showing hit rate.

    Args:
        ax: Matplotlib axes to draw on
        hit_rate: Hit rate percentage
    """
    # Set up gauge parameters
    gauge_min = 0
    gauge_max = 100
    gauge_range = gauge_max - gauge_min

    # Create gauge background
    theta = np.linspace(np.pi, 0, 100)
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Draw gauge background
    ax.plot(x, y, color='lightgray', linewidth=10, solid_capstyle='round')

    # Calculate hit rate position
    hit_rate_normalized = max(0, min(1, hit_rate / gauge_range))
    hit_theta = np.linspace(np.pi, np.pi - hit_rate_normalized * np.pi, 100)
    hit_x = r * np.cos(hit_theta)
    hit_y = r * np.sin(hit_theta)

    # Determine color based on hit rate
    if hit_rate < 20:
        color = 'red'
    elif hit_rate < 40:
        color = 'orange'
    elif hit_rate < 60:
        color = 'yellow'
    else:
        color = 'green'

    # Draw hit rate gauge segment
    ax.plot(hit_x, hit_y, color=color, linewidth=10, solid_capstyle='round')

    # Add hit rate text
    ax.text(0, 0, f"{hit_rate:.1f}%", ha='center', va='center', fontsize=24,
            fontweight='bold', color=THEME_COLORS['text'])
    ax.text(0, -0.3, "Hit Rate", ha='center', va='center', fontsize=14,
            color=THEME_COLORS['text'])

    # Add gauge ticks and labels
    for val, label in [(0, '0%'), (25, '25%'), (50, '50%'), (75, '75%'), (100, '100%')]:
        val_normalized = val / gauge_range
        tick_theta = np.pi - val_normalized * np.pi
        tick_x = 0.9 * np.cos(tick_theta)
        tick_y = 0.9 * np.sin(tick_theta)
        tick_x2 = 0.82 * np.cos(tick_theta)
        tick_y2 = 0.82 * np.sin(tick_theta)
        ax.plot([tick_x, tick_x2], [tick_y, tick_y2], color='gray', linewidth=1.5)
        ax.text(
            0.75 * np.cos(tick_theta),
            0.75 * np.sin(tick_theta),
            label,
            ha='center', va='center',
            fontsize=8, color='gray'
        )

    # Add interpretation text
    if hit_rate < 20:
        interpretation = "Poor"
    elif hit_rate < 40:
        interpretation = "Fair"
    elif hit_rate < 60:
        interpretation = "Good"
    elif hit_rate < 80:
        interpretation = "Very Good"
    else:
        interpretation = "Excellent"

    ax.text(0, -0.6, f"Performance: {interpretation}", ha='center', va='center',
            fontsize=10, fontweight='bold', color=color)

    # Set up axes
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Prediction Performance', fontsize=14, pad=10)


def _plot_top_files(ax: plt.Axes, top_files: List[Dict[str, Any]], max_files: int = 7) -> None:
    """
    Plot top bug-prone files.

    Args:
        ax: Matplotlib axes to draw on
        top_files: List of top files with bug info
        max_files: Maximum number of files to show
    """
    # Prepare data
    if not top_files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Top Bug-Prone Files', fontsize=14)
        ax.axis('off')
        return

    # Limit to max_files
    if len(top_files) > max_files:
        files_to_plot = top_files[:max_files]
    else:
        files_to_plot = top_files

    # Extract file names and bug counts
    file_names = [os.path.basename(file_info['file_path']) for file_info in files_to_plot]
    bug_counts = [file_info['bug_fixes'] for file_info in files_to_plot]

    # Create horizontal bar chart
    y_pos = np.arange(len(file_names))
    ax.barh(y_pos, bug_counts, color=THEME_COLORS['primary'], alpha=0.8)

    # Add file names and bug counts
    for i, count in enumerate(bug_counts):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=10,
                fontweight='bold', color=THEME_COLORS['text'])

    # Add file extensions as colored markers
    for i, file_info in enumerate(files_to_plot):
        ext = os.path.splitext(file_info['file_path'])[1]
        if ext:
            ax.text(-0.8, i, ext, va='center', ha='center', fontsize=8,
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=THEME_COLORS['secondary'],
                              alpha=0.8, boxstyle='round,pad=0.2'))

    # Set up axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(file_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Number of Bug Fixes', fontsize=10)
    ax.set_title('Top Bug-Prone Files', fontsize=14)

    # Add grid lines for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_bug_distribution_by_type(ax: plt.Axes, files: List[Dict[str, Any]]) -> None:
    """
    Plot bug distribution by file type.

    Args:
        ax: Matplotlib axes to draw on
        files: List of files with bug info
    """
    # Check for data
    if not files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Bug Distribution by File Type', fontsize=14)
        ax.axis('off')
        return

    # Extract file extensions and count bugs
    ext_bugs = defaultdict(int)
    for file_info in files:
        ext = os.path.splitext(file_info['file_path'])[1]
        if not ext:
            ext = 'unknown'
        ext_bugs[ext] += file_info['bug_fixes']

    # Sort by bug count
    ext_bugs = dict(sorted(ext_bugs.items(), key=lambda x: x[1], reverse=True))

    # Combine small categories into "Other" if many file types
    if len(ext_bugs) > 6:
        main_exts = dict(list(ext_bugs.items())[:5])
        other_count = sum(list(ext_bugs.values())[5:])
        if other_count > 0:
            main_exts['Other'] = other_count
        ext_bugs = main_exts

    # Create data for plotting
    extensions = list(ext_bugs.keys())
    bug_counts = list(ext_bugs.values())
    total_bugs = sum(bug_counts)

    # Generate colors for the pie chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(extensions)))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        bug_counts,
        labels=None,
        autopct=None,
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True}
    )

    # Add percentage labels inside pie
    for i, autotext in enumerate(autotexts):
        percentage = bug_counts[i] / total_bugs * 100
        autotext.set_text(f"{percentage:.1f}%")
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Create legend with file extensions and counts
    legend_labels = [f"{ext} ({count} bugs, {count / total_bugs * 100:.1f}%)"
                     for ext, count in zip(extensions, bug_counts)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(-0.1, 0, 0.5, 1),
              fontsize=9)

    # Add title
    ax.set_title('Bug Distribution by File Type', fontsize=14)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')


def _plot_bug_timeline(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot bug timeline if available in results.

    Args:
        ax: Matplotlib axes to draw on
        results: Results dictionary
    """
    # Check if we have time distribution data
    if 'temporal_distribution' not in results:
        # Simulate time data if not available
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        bug_counts = np.random.randint(1, 10, size=12)

        # Plot simulated data
        ax.plot(months, bug_counts, marker='o', linestyle='-', color=THEME_COLORS['tertiary'],
                linewidth=2, markersize=8)

        # Add area under the curve
        ax.fill_between(months, bug_counts, alpha=0.3, color=THEME_COLORS['tertiary'])

        # Set up axes
        ax.set_xlabel('Time (Months)', fontsize=10)
        ax.set_ylabel('Bug Fixes', fontsize=10)
        ax.set_title('Bug Fix Timeline (Simulated)', fontsize=14)

        # Add note about simulated data
        ax.text(0.5, 0.95, "Note: Using simulated data for demonstration",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, fontstyle='italic', color='gray')
    else:
        # Use real temporal distribution data
        temporal = results['temporal_distribution']

        # Plot real data
        ax.plot(temporal['labels'], temporal['counts'], marker='o', linestyle='-',
                color=THEME_COLORS['tertiary'], linewidth=2, markersize=8)

        # Add area under the curve
        ax.fill_between(temporal['labels'], temporal['counts'], alpha=0.3,
                        color=THEME_COLORS['tertiary'])

        # Set up axes
        period_type = temporal.get('period_type', 'time')
        ax.set_xlabel(f'Time ({period_type.capitalize()}s)', fontsize=10)
        ax.set_ylabel('Bug Fixes', fontsize=10)
        ax.set_title('Bug Fix Timeline', fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_file_complexity(ax: plt.Axes, files: List[Dict[str, Any]]) -> None:
    """
    Plot file complexity vs bug count.

    Args:
        ax: Matplotlib axes to draw on
        files: List of files with bug info
    """
    # Check for data
    if not files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('File Complexity vs. Bug Count', fontsize=14)
        ax.axis('off')
        return

    # Extract data - we'll use file type as a proxy for complexity group
    file_types = {}
    for file_info in files:
        ext = os.path.splitext(file_info['file_path'])[1]
        if not ext:
            ext = 'unknown'

        if ext not in file_types:
            file_types[ext] = []

        file_types[ext].append({
            'name': os.path.basename(file_info['file_path']),
            'bugs': file_info['bug_fixes']
        })

    # Set up colors for different file types
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_types)))
    color_map = {ext: colors[i] for i, ext in enumerate(file_types.keys())}

    # Plot each file as a bubble
    for ext, files in file_types.items():
        # Extract x and y coordinates for this file type
        x = np.arange(len(files))
        y = [f['bugs'] for f in files]
        sizes = [f['bugs'] * 50 for f in files]  # Scale bubble size

        # Plot bubbles
        scatter = ax.scatter(
            x, y, s=sizes, c=[color_map[ext]],
            alpha=0.7, edgecolors='w', linewidth=0.5,
            label=ext
        )

        # Add file names as annotations
        for i, file in enumerate(files):
            if file['bugs'] > 1:  # Only label significant files
                ax.annotate(
                    file['name'],
                    (x[i], y[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

    # Create legend
    legend = ax.legend(
        title="File Types",
        loc="upper right",
        fontsize=9
    )
    legend.get_title().set_fontsize(10)

    # Set up axes
    ax.set_xlabel('Files Grouped by Type', fontsize=10)
    ax.set_ylabel('Number of Bug Fixes', fontsize=10)
    ax.set_title('File Type vs. Bug Count', fontsize=14)

    # Remove actual x-ticks as they're not meaningful in this context
    ax.set_xticks([])

    # Add grid for y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _plot_hit_miss_pie(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot hit/miss distribution as a pie chart.

    Args:
        ax: Matplotlib axes to draw on
        results: Results dictionary
    """
    # Extract hit and miss counts
    hit_count = results.get('hit_count', 0)
    miss_count = results.get('miss_count', 0)

    # Check for data
    if hit_count == 0 and miss_count == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Hit/Miss Distribution', fontsize=14)
        ax.axis('off')
        return

    # Prepare data for pie chart
    labels = ['Hits', 'Misses']
    sizes = [hit_count, miss_count]
    colors = [THEME_COLORS['tertiary'], THEME_COLORS['quaternary']]
    explode = (0.05, 0)  # Explode the first slice (Hits)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=None,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True}
    )

    # Customize autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    # Add legend
    legend_labels = [f"{label} ({size})" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='center', fontsize=9)

    # Add title
    ax.set_title('Hit/Miss Distribution', fontsize=14)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')


def plot_cache_optimization(results, output_file):
    """
    Generate visualization for cache size optimization.

    Args:
        results (dict): Dictionary mapping cache sizes to hit rates
        output_file (str): The path to save the visualization

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Convert string keys to float if needed
        cache_sizes = []
        hit_rates = []

        for size, rate in sorted(results.items()):
            # Convert to float if it's a string
            if isinstance(size, str):
                size = float(size)
            cache_sizes.append(size * 100)  # Convert to percentage
            hit_rates.append(rate)

        # Find best cache size
        best_idx = hit_rates.index(max(hit_rates))
        best_size = cache_sizes[best_idx]
        best_rate = hit_rates[best_idx]

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot cache size vs hit rate
        plt.plot(cache_sizes, hit_rates, 'o-', linewidth=2, color='#2196F3')

        # Highlight the optimal point
        plt.plot(best_size, best_rate, 'ro', markersize=10)
        plt.annotate('Optimal: {:.1f}%'.format(best_size),
                     xy=(best_size, best_rate),
                     xytext=(best_size + 2, best_rate - 5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     )

        # Set labels and title
        plt.xlabel('Cache Size (% of total files)')
        plt.ylabel('Hit Rate (%)')
        plt.title('Cache Size Optimization')

        # Set grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info("Optimization visualization successfully saved to {}".format(output_file))
        return True

    except ImportError:
        logger.error("Matplotlib is required for visualization. Install with: pip install matplotlib")
        return False
    except Exception as e:
        logger.error("Error generating visualization: {}".format(str(e)))
        return False


def plot_repository_comparison(results, output_file):
    """
    Generate visualization comparing multiple repositories.

    Args:
        results (dict): Dictionary mapping repository names to their results
        output_file (str): The path to save the visualization

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract data
        repo_names = list(results.keys())
        hit_rates = []
        bug_fix_ratios = []

        for repo, data in results.items():
            hit_rates.append(data.get('hit_rate', 0))
            total_commits = data.get('total_commits', 0)
            bug_fixes = data.get('bug_fixes', 0)
            bug_fix_ratio = (bug_fixes / total_commits * 100) if total_commits > 0 else 0
            bug_fix_ratios.append(bug_fix_ratio)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar width
        width = 0.35

        # Plot 1: Hit Rates
        x = np.arange(len(repo_names))
        ax1.bar(x, hit_rates, width, color='#4CAF50', label='Hit Rate')

        # Set labels and title
        ax1.set_xlabel('Repository')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.set_title('Prediction Hit Rate by Repository')
        ax1.set_xticks(x)
        ax1.set_xticklabels(repo_names, rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Add text labels
        for i, v in enumerate(hit_rates):
            ax1.text(i, v + 1, "{:.1f}%".format(v), ha='center')

        # Plot 2: Bug Fix Ratios
        ax2.bar(x, bug_fix_ratios, width, color='#F44336', label='Bug Fix Ratio')

        # Set labels and title
        ax2.set_xlabel('Repository')
        ax2.set_ylabel('Bug Fixes (% of total commits)')
        ax2.set_title('Bug Fix Ratio by Repository')
        ax2.set_xticks(x)
        ax2.set_xticklabels(repo_names, rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Add text labels
        for i, v in enumerate(bug_fix_ratios):
            ax2.text(i, v + 1, "{:.1f}%".format(v), ha='center')

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info("Repository comparison visualization successfully saved to {}".format(output_file))
        return True

    except ImportError:
        logger.error("Matplotlib is required for visualization. Install with: pip install matplotlib")
        return False
    except Exception as e:
        logger.error("Error generating visualization: {}".format(str(e)))
        return False