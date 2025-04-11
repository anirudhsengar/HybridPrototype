#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Command Line Interface

This module provides the command-line interface for the FixCache tool.

Author: anirudhsengar
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from . import __version__
from .algorithm import FixCache
from .repository import RepositoryAnalyzer
from .visualization import visualize_results, plot_cache_optimization, plot_repository_comparison


def setup_logging(verbose=False, quiet=False):
    """Set up logging configuration."""
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('fixcache')


def analyze_command(args):
    """Handle the analyze command."""
    logger = setup_logging(args.verbose, args.quiet)

    logger.info("Analyzing repository: {}".format(args.repo_path))

    # Create FixCache instance
    try:
        fix_cache = FixCache(
            repo_path=args.repo_path,
            cache_size=args.cache_size,
            policy=args.policy,
            window_ratio=args.window_ratio,
            lookback_commits=args.lookback if args.lookback else None,
            cache_seeding=not args.no_seeding
        )

        # Analyze repository
        logger.info("Analyzing repository history...")
        analyze_success = fix_cache.analyze_repository()

        if not analyze_success:
            logger.error("Repository analysis failed. Check error messages.")
            return 1

        # Run prediction
        logger.info("Running prediction...")
        hit_rate = fix_cache.predict()

        # Get summary
        results = fix_cache.get_summary()

        # Display results
        logger.info("Hit rate: {:.2f}%".format(hit_rate))
        logger.info("Cache size: {:.1f}% of {} files".format(args.cache_size * 100, results.get('total_files', 'N/A')))

        # Determine output files
        viz_output = None
        report_output = None
        data_output = None

        if args.visualize and args.output and args.output.endswith(('.png', '.jpg', '.pdf')):
            viz_output = args.output
        elif args.visualize:
            viz_output = "fixcache_results.png"

        if args.report and args.output and not args.output.endswith(('.png', '.jpg', '.pdf')):
            report_output = args.output
        elif args.report:
            report_output = "fixcache_report.md"

        if not args.visualize and not args.report and args.output:
            data_output = args.output

        # Generate visualization if requested
        if args.visualize:
            try:
                logger.info("Generating visualization to {}".format(viz_output))
                from .visualization import visualize_results
                visualize_success = visualize_results(results, viz_output)
                if not visualize_success:
                    logger.warning("Failed to generate visualization. Check if matplotlib is installed.")
            except Exception as e:
                logger.error("Error generating visualization: {}".format(str(e)))
                logger.warning("Failed to generate visualization. Matplotlib may not be installed.")

        # Generate report if requested
        if args.report:
            try:
                format_type = args.format if args.format else "md"
                logger.info("Generating report to {}".format(report_output))

                report_content = generate_report(fix_cache, format_type)
                with open(report_output, 'w') as f:
                    f.write(report_content)

                logger.info("Report successfully generated")
            except Exception as e:
                logger.error("Error generating report: {}".format(str(e)))

        # Output results in the requested format
        if data_output:
            format_type = args.format if args.format else "json"
            logger.info("Writing results to {}".format(data_output))

            try:
                if format_type == "json":
                    with open(data_output, 'w') as f:
                        json.dump(results, f, indent=2)
                elif format_type == "yaml":
                    try:
                        import yaml
                        with open(data_output, 'w') as f:
                            yaml.dump(results, f)
                    except ImportError:
                        logger.error("YAML output requires PyYAML. Install with: pip install pyyaml")
                        return 1
                else:
                    with open(data_output, 'w') as f:
                        f.write(str(results))
            except Exception as e:
                logger.error("Error writing results: {}".format(str(e)))
                return 1

        return 0

    except Exception as e:
        logger.error("Error during analysis: {}".format(str(e)))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def generate_report(fix_cache, format_type):
    """
    Generate a report based on FixCache analysis results.

    Args:
        fix_cache: The FixCache instance with analysis results
        format_type: The format for the report (md, txt, etc.)

    Returns:
        str: The formatted report content
    """
    results = fix_cache.get_summary() if hasattr(fix_cache, 'get_summary') else {}
    top_files = fix_cache.get_top_files(20) if hasattr(fix_cache, 'get_top_files') else []

    # Ensure repository_path exists
    repo_path = getattr(fix_cache, 'repo_path', 'Unknown Repository')
    if 'repository_path' not in results or not results['repository_path']:
        results['repository_path'] = repo_path

    if format_type == "md":
        report = "# FixCache Analysis Report\n\n"
        report += "## Repository Summary\n\n"
        report += "- **Repository Path**: {}\n".format(results.get('repository_path', 'Unknown Repository'))
        report += "- **Analysis Date**: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report += "- **Total Files**: {}\n".format(results.get('total_files', 'N/A'))
        report += "- **Total Commits**: {}\n".format(results.get('total_commits', 'N/A'))
        report += "- **Bug Fixes**: {}\n\n".format(results.get('bug_fixes', 'N/A'))

        report += "## Analysis Parameters\n\n"
        cache_size = results.get('cache_size', 0)
        if isinstance(cache_size, str):
            try:
                cache_size = float(cache_size)
            except (ValueError, TypeError):
                cache_size = 0
        report += "- **Cache Size**: {:.1f}%\n".format(cache_size * 100)
        report += "- **Policy**: {}\n".format(results.get('policy', 'N/A'))
        report += "- **Window Ratio**: {}\n\n".format(results.get('window_ratio', 'N/A'))

        report += "## Results\n\n"
        hit_rate = results.get('hit_rate', 0)
        if isinstance(hit_rate, str):
            try:
                hit_rate = float(hit_rate)
            except (ValueError, TypeError):
                hit_rate = 0
        report += "- **Hit Rate**: {:.2f}%\n\n".format(hit_rate)

        if top_files:
            report += "## Top Bug-Prone Files\n\n"
            report += "| Rank | File | Risk Score |\n"
            report += "|------|------|------------|\n"
            for i, (file, score) in enumerate(top_files, 1):
                # Ensure score is a float
                try:
                    score_float = float(score)
                except (ValueError, TypeError):
                    score_float = 0.0
                report += "| {} | {} | {:.2f} |\n".format(i, file, score_float)

    elif format_type == "txt":
        report = "FixCache Analysis Report\n"
        report += "=======================\n\n"
        report += "Repository Path: {}\n".format(results.get('repository_path', 'Unknown Repository'))
        report += "Analysis Date: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report += "Total Files: {}\n".format(results.get('total_files', 'N/A'))
        report += "Total Commits: {}\n".format(results.get('total_commits', 'N/A'))
        report += "Bug Fixes: {}\n\n".format(results.get('bug_fixes', 'N/A'))

        report += "Analysis Parameters:\n"
        cache_size = results.get('cache_size', 0)
        if isinstance(cache_size, str):
            try:
                cache_size = float(cache_size)
            except (ValueError, TypeError):
                cache_size = 0
        report += "  Cache Size: {:.1f}%\n".format(cache_size * 100)
        report += "  Policy: {}\n".format(results.get('policy', 'N/A'))
        report += "  Window Ratio: {}\n\n".format(results.get('window_ratio', 'N/A'))

        report += "Results:\n"
        hit_rate = results.get('hit_rate', 0)
        if isinstance(hit_rate, str):
            try:
                hit_rate = float(hit_rate)
            except (ValueError, TypeError):
                hit_rate = 0
        report += "  Hit Rate: {:.2f}%\n\n".format(hit_rate)

        if top_files:
            report += "Top Bug-Prone Files:\n"
            for i, (file, score) in enumerate(top_files, 1):
                # Ensure score is a float
                try:
                    score_float = float(score)
                except (ValueError, TypeError):
                    score_float = 0.0
                report += "{}. {} (Risk Score: {:.2f})\n".format(i, file, score_float)

    else:
        # Default to JSON string
        import json
        report_data = {
            "repository": results.get('repository_path', 'Unknown Repository'),
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "parameters": {
                "cache_size": results.get('cache_size', 'N/A'),
                "policy": results.get('policy', 'N/A'),
                "window_ratio": results.get('window_ratio', 'N/A')
            },
            "results": {
                "total_files": results.get('total_files', 'N/A'),
                "total_commits": results.get('total_commits', 'N/A'),
                "bug_fixes": results.get('bug_fixes', 'N/A'),
                "hit_rate": results.get('hit_rate', 'N/A')
            },
            "top_files": [{"file": file, "score": float(score) if isinstance(score, (int, float)) else 0.0}
                          for file, score in top_files]
        }
        report = json.dumps(report_data, indent=2)

    return report


def optimize_command(args):
    """Handle the optimize command."""
    logger = setup_logging(args.verbose, args.quiet)

    logger.info("Optimizing cache size for repository: {}".format(args.repo_path))

    try:
        # Create cache sizes to test
        cache_sizes = []
        current_size = args.min_size
        while current_size <= args.max_size:
            cache_sizes.append(round(current_size, 4))
            current_size += args.step_size

        logger.info("Testing cache sizes: {}".format(cache_sizes))

        # Initialize results dictionary
        results = {}

        # Test each cache size
        for size in cache_sizes:
            logger.info("Testing cache size: {:.1f}%".format(size * 100))

            fix_cache = FixCache(
                repo_path=args.repo_path,
                cache_size=size,
                policy=args.policy,
                lookback_commits=args.lookback if args.lookback else None,  # Use lookback parameter
                window_ratio=0.25  # Use default window ratio for optimization
            )

            analyze_success = fix_cache.analyze_repository()
            if not analyze_success:
                logger.error("Repository analysis failed. Check error messages.")
                return 1

            hit_rate = fix_cache.predict()
            results[size] = hit_rate

            logger.info("Cache size {:.1f}% hit rate: {:.2f}%".format(size * 100, hit_rate))

        # Find best cache size
        best_size = max(results, key=results.get)
        best_hit_rate = results[best_size]

        logger.info("Optimal cache size: {:.1f}% with hit rate: {:.2f}%".format(best_size * 100, best_hit_rate))

        # Generate visualization if output is provided
        if args.output:
            if args.output.endswith('.png'):
                logger.info("Generating visualization to {}".format(args.output))
                from .visualization import plot_cache_optimization
                plot_cache_optimization(results, args.output)
            else:
                logger.info("Writing results to {}".format(args.output))
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

        return 0

    except Exception as e:
        logger.error("Error during optimization: {}".format(str(e)))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def compare_command(args):
    """Handle the compare command."""
    logger = setup_logging(args.verbose, args.quiet)

    repos = args.repos.split(',')
    logger.info(f"Comparing {len(repos)} repositories")

    try:
        results = {}

        for repo_path in repos:
            repo_path = repo_path.strip()
            logger.info(f"Analyzing repository: {repo_path}")

            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=args.cache_size,
                policy=args.policy,
                lookback_commits=args.lookback if args.lookback else None  # Use lookback parameter
            )

            analyze_success = fix_cache.analyze_repository()
            if not analyze_success:
                logger.warning(f"Repository analysis failed for {repo_path}. Skipping.")
                continue

            hit_rate = fix_cache.predict()
            summary = fix_cache.get_summary()

            repo_name = os.path.basename(os.path.abspath(repo_path))
            results[repo_name] = summary

            logger.info(f"{repo_name}: hit rate {hit_rate:.2f}%")

        # Generate comparison visualization if output is provided
        if args.output:
            if args.output.endswith('.png'):
                logger.info(f"Generating comparison visualization to {args.output}")
                from .visualization import plot_repository_comparison
                plot_repository_comparison(results, args.output)
            else:
                logger.info(f"Writing comparison results to {args.output}")
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

        return 0

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def visualize_command(args):
    """Handle the visualize command."""
    logger = setup_logging(args.verbose, args.quiet)

    try:
        # Load input file
        logger.info(f"Loading results from {args.input}")
        with open(args.input, 'r') as f:
            results = json.load(f)

        # Generate visualization
        output_file = args.output if args.output else "fixcache_visualization.png"
        logger.info(f"Generating visualization to {output_file}")

        try:
            if args.type == "summary":
                from .visualization import visualize_results
                visualize_success = visualize_results(results, output_file)
            elif args.type == "optimization" and isinstance(results, dict):
                from .visualization import plot_cache_optimization
                visualize_success = plot_cache_optimization(results, output_file)
            elif args.type == "comparison":
                from .visualization import plot_repository_comparison
                visualize_success = plot_repository_comparison(results, output_file)
            else:
                logger.error(f"Unsupported visualization type: {args.type}")
                return 1
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            return 1

        if not visualize_success:
            logger.warning("Failed to generate visualization. Check if matplotlib is installed.")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="FixCache - A tool for bug prediction using repository history"
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Increase output verbosity'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true', help='Suppress output'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a repository')
    analyze_parser.add_argument('repo_path', help='Path to the repository to analyze')
    analyze_parser.add_argument(
        '--cache-size', '-c', type=float, default=0.2,
        help='Cache size as fraction of total files (default: 0.2)'
    )
    analyze_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    analyze_parser.add_argument(
        '--window-ratio', '-w', type=float, default=0.25,
        help='History window ratio (default: 0.25)'
    )
    analyze_parser.add_argument(
        '--lookback', '-l', type=int,
        help='Number of commits to look back (default: all)'
    )
    analyze_parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization'
    )
    analyze_parser.add_argument(
        '--report', '-r', action='store_true',
        help='Generate report'
    )
    analyze_parser.add_argument(
        '--no-seeding', action='store_true',
        help='Disable cache seeding'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    analyze_parser.add_argument(
        '--format', '-f', choices=['json', 'yaml', 'md', 'txt'], default='json',
        help='Output format (default: json)'
    )
    analyze_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Increase output verbosity'
    )
    analyze_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache size')
    optimize_parser.add_argument('repo_path', help='Path to the repository to analyze')
    optimize_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    optimize_parser.add_argument(
        '--min-size', type=float, default=0.05,
        help='Minimum cache size to test (default: 0.05)'
    )
    optimize_parser.add_argument(
        '--max-size', type=float, default=0.5,
        help='Maximum cache size to test (default: 0.5)'
    )
    optimize_parser.add_argument(
        '--step-size', type=float, default=0.05,
        help='Step size for cache testing (default: 0.05)'
    )
    optimize_parser.add_argument(
        '--lookback', '-l', type=int,
        help='Number of commits to look back (default: all)'
    )
    optimize_parser.add_argument(
        '--fine-grained', action='store_true',
        help='Use fine-grained steps'
    )
    optimize_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    optimize_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Increase output verbosity'
    )
    optimize_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare repositories')
    compare_parser.add_argument(
        '--repos', '-r', required=True,
        help='Comma-separated list of repository paths'
    )
    compare_parser.add_argument(
        '--cache-size', '-c', type=float, default=0.2,
        help='Cache size to use for comparison (default: 0.2)'
    )
    compare_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    compare_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    compare_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Increase output verbosity'
    )
    compare_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )
    compare_parser.add_argument(
        '--lookback', '-l', type=int,
        help='Number of commits to look back (default: all)'
    )

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    visualize_parser.add_argument(
        '--input', '-i', required=True,
        help='Input JSON file with analysis results'
    )
    visualize_parser.add_argument(
        '--type', '-t', choices=['summary', 'optimization', 'comparison'], default='summary',
        help='Visualization type (default: summary)'
    )
    visualize_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    visualize_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Increase output verbosity'
    )
    visualize_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'optimize':
        return optimize_command(args)
    elif args.command == 'compare':
        return compare_command(args)
    elif args.command == 'visualize':
        return visualize_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())