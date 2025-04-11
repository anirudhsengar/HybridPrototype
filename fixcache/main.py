#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Main Entry Point

Command-line interface and main entry point for the FixCache bug prediction tool.

Author: anirudhsengar
"""

import os
import sys
import argparse
import logging
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import components from the package
from .algorithm import FixCache
from .utils import optimize_cache_size, compare_repositories
from .visualization import visualize_results, plot_cache_optimization
from fixcache import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fixcache')


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='FixCache: Enhanced Bug Prediction Tool',
        epilog=(
            'Example usage:\n'
            '  # Run basic prediction:\n'
            '  fixcache --repo-path /path/to/repo --output-file results.json\n'
            '  # Optimize cache size:\n'
            '  fixcache --repo-path /path/to/repo --optimize --output-file results.json\n'
            '  # Compare repositories:\n'
            '  fixcache --compare repo1,repo2,repo3'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Main operation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--optimize', action='store_true',
        help='Run cache size optimization'
    )
    mode_group.add_argument(
        '--compare', metavar='REPOS',
        help='Compare multiple repositories (comma-separated paths)'
    )

    # Repository options
    parser.add_argument(
        '--repo-path', metavar='PATH',
        help='Path to Git repository'
    )
    parser.add_argument(
        '--lookback', type=int, metavar='N',
        help='Number of recent commits to analyze (default: all)'
    )
    parser.add_argument(
        '--small-repo', action='store_true',
        help='Optimize for small repositories'
    )

    # FixCache algorithm options
    parser.add_argument(
        '--cache-size', type=float, default=0.2, metavar='RATIO',
        help='Cache size as fraction of total files (default: 0.2)'
    )
    parser.add_argument(
        '--policy', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    parser.add_argument(
        '--window-ratio', type=float, default=0.25, metavar='RATIO',
        help='Ratio of commits to use for training window (default: 0.25)'
    )
    parser.add_argument(
        '--no-cache-seeding', action='store_true',
        help='Disable cache seeding with most bug-prone files'
    )

    # Output options
    parser.add_argument(
        '--output-file', metavar='FILE',
        help='Output file for results (default: fixcache_results.json)'
    )
    parser.add_argument(
        '--no-visualization', action='store_true',
        help='Disable visualization generation'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress non-error output'
    )
    parser.add_argument(
        '--version', action='version',
        version=f'FixCache {__version__}'
    )

    return parser


def run_standard_prediction(args: argparse.Namespace) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Run standard FixCache prediction.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (success status, results dict)
    """
    logger.info(f"Running standard prediction on {args.repo_path}")

    # Create FixCache instance
    fix_cache = FixCache(
        repo_path=args.repo_path,
        cache_size=args.cache_size,
        policy=args.policy,
        lookback_commits=args.lookback,
        window_ratio=args.window_ratio,
        cache_seeding=not args.no_cache_seeding
    )

    # Run repository analysis
    if not fix_cache.analyze_repository():
        logger.error(f"Repository analysis failed: {'; '.join(fix_cache.error_messages)}")
        return False, None

    # Run prediction
    hit_rate = fix_cache.predict()

    # Create default output file name if not provided
    output_file = args.output_file or f"fixcache_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Save results
    fix_cache.save_results(output_file)

    # Generate visualization
    if not args.no_visualization:
        try:
            visualization_file = os.path.splitext(output_file)[0] + '.png'
            fix_cache.visualize_results(visualization_file)
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")

    # Print summary
    if not args.quiet:
        repo_name = os.path.basename(os.path.abspath(args.repo_path))
        print(f"\n{'-' * 60}")
        print(f"FixCache Prediction Results for {repo_name}")
        print(f"{'-' * 60}")
        print(f"Total files: {fix_cache.repo_analyzer.total_files}")
        print(f"Bug-fixing commits: {len(fix_cache.repo_analyzer.bug_fixes)}")
        print(f"Cache size: {args.cache_size * 100}% ({fix_cache.cache_max_size} files)")
        print(f"Hit rate: {hit_rate:.2f}%")
        print(f"Results saved to: {output_file}")

        # Print top files
        print("\nTop files most likely to contain bugs:")
        for i, file_info in enumerate(fix_cache.results['top_files'][:5]):
            print(f"{i + 1}. {file_info['file_path']} - {file_info['bug_fixes']} bug fixes")
        print()

        # Print recommendations
        recommendations = fix_cache.get_recommended_actions()
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"- {rec['message']}")
            print()

    return True, fix_cache.results


def run_optimization(args: argparse.Namespace) -> Tuple[bool, Optional[Dict[float, float]]]:
    """
    Run cache size optimization.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (success status, results dict)
    """
    logger.info(f"Running cache size optimization on {args.repo_path}")

    # Determine cache sizes to test
    if args.small_repo:
        # For small repos, use smaller increments and range
        cache_sizes = [0.05, 0.1, 0.15, 0.2, 0.25]
    else:
        # Standard range for regular repos
        cache_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]

    # Create output file prefix
    output_prefix = args.output_file or f"fixcache_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_prefix = os.path.splitext(output_prefix)[0]

    # Run optimization
    try:
        results = optimize_cache_size(
            args.repo_path,
            output_prefix,
            cache_sizes,
            policy=args.policy,
            lookback_commits=args.lookback,
            window_ratio=args.window_ratio,
            cache_seeding=not args.no_cache_seeding
        )

        # Print summary
        if not args.quiet and results:
            repo_name = os.path.basename(os.path.abspath(args.repo_path))
            print(f"\n{'-' * 60}")
            print(f"FixCache Cache Size Optimization for {repo_name}")
            print(f"{'-' * 60}")
            print("Cache Size Optimization Results:")

            for size, hit_rate in sorted(results.items()):
                print(f"  Cache Size {size * 100:.1f}%: Hit Rate {hit_rate:.2f}%")

            # Find optimal cache size
            optimal_size = max(results.items(), key=lambda x: x[1])[0]
            print(f"\nOptimal cache size: {optimal_size * 100:.1f}%")
            print(f"Results saved to: {output_prefix}_*.json")
            print()

        return True, results

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return False, None


def run_comparison(args: argparse.Namespace) -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
    """
    Run repository comparison.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (success status, results dict)
    """
    # Split repository paths
    repo_paths = [p.strip() for p in args.compare.split(',')]

    if len(repo_paths) < 2:
        logger.error("At least two repositories must be specified for comparison")
        return False, None

    logger.info(f"Comparing {len(repo_paths)} repositories")

    # Create output file
    output_file = args.output_file or f"fixcache_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Run comparison
    try:
        results = compare_repositories(
            repo_paths,
            cache_size=args.cache_size,
            policy=args.policy,
            lookback_commits=args.lookback,
            output_file=output_file,
            visualization=(not args.no_visualization)
        )

        # Print summary
        if not args.quiet and results:
            print(f"\n{'-' * 60}")
            print(f"FixCache Repository Comparison")
            print(f"{'-' * 60}")
            print("Comparison Results:")

            # Format as table
            headers = ["Repository", "Files", "Bug Fixes", "Cache Size", "Hit Rate"]
            rows = []

            for repo_path, repo_results in results.items():
                repo_name = os.path.basename(os.path.abspath(repo_path))
                rows.append([
                    repo_name,
                    str(repo_results.get('total_files', 'N/A')),
                    str(repo_results.get('total_bug_fixes', 'N/A')),
                    f"{args.cache_size * 100:.1f}%",
                    f"{repo_results.get('hit_rate', 0):.2f}%"
                ])

            # Get column widths
            col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) + 2
                          for i in range(len(headers))]

            # Print headers
            header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(header_row)
            print("-" * len(header_row))

            # Print rows
            for row in rows:
                print("".join(cell.ljust(w) for cell, w in zip(row, col_widths)))

            print(f"\nDetailed results saved to: {output_file}")
            print()

        return True, results

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        return False, None


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the FixCache command-line tool.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = setup_argument_parser()
    parsed_args = parser.parse_args(args)

    # Configure logging
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    elif parsed_args.quiet:
        logger.setLevel(logging.ERROR)

    # Validate arguments
    if parsed_args.compare:
        # Repository comparison mode
        if not parsed_args.compare:
            parser.error("Repository paths must be specified with --compare")
            return 1
    else:
        # Standard or optimization mode
        if not parsed_args.repo_path:
            parser.error("Repository path must be specified with --repo-path")
            return 1

        # Check if repository exists
        if not os.path.exists(parsed_args.repo_path):
            logger.error(f"Repository path does not exist: {parsed_args.repo_path}")
            return 1

        # Check if repository is a git repository
        git_dir = os.path.join(parsed_args.repo_path, '.git')
        if not os.path.exists(git_dir):
            logger.error(f"Not a git repository: {parsed_args.repo_path}")
            return 1

    try:
        # Record start time
        start_time = datetime.datetime.now()

        # Run the appropriate mode
        if parsed_args.compare:
            success, results = run_comparison(parsed_args)
        elif parsed_args.optimize:
            success, results = run_optimization(parsed_args)
        else:
            success, results = run_standard_prediction(parsed_args)

        # Calculate and display elapsed time
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        if not parsed_args.quiet:
            print(f"Total execution time: {elapsed_time:.2f} seconds")

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=parsed_args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())