#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Functions for FixCache

This module provides utility functions for the FixCache algorithm,
including cache size optimization, repository comparison, and helper functions.

Author: anirudhsengar
"""

import os
import re
import logging
import json
import datetime
from typing import List, Dict, Set, Tuple, Any, Optional, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import visualizations (avoid circular imports by importing inside functions)
# from .visualization import plot_cache_optimization, plot_repository_comparison

# Setup logging
logger = logging.getLogger(__name__)


def safe_divide(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Safely perform division, avoiding division by zero errors.

    Args:
        numerator: Numerator value
        denominator: Denominator value

    Returns:
        Result of division, or 0 if denominator is 0
    """
    try:
        if denominator == 0:
            return 0
        return numerator / denominator
    except:
        return 0


def is_code_file(file_path: str) -> bool:
    """
    Check if a file is a code file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        True if it's a code file, False otherwise
    """
    # List of common code file extensions
    code_extensions = [
        '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.js', '.ts',
        '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.rs', '.sh',
        '.xml', '.gradle', '.properties', '.cmake', '.mk', '.m', '.json',
        '.html', '.css', '.sql', '.groovy', '.bat', '.yaml', '.yml',
        '.jsx', '.tsx', '.md', '.R', '.pl', '.dart', '.config'
    ]

    # Ignore certain paths
    ignore_patterns = [
        r'(^|/)\.git/',  # Git directory
        r'(^|/)node_modules/',  # Node.js modules
        r'(^|/)build/',  # Build directory
        r'(^|/)dist/',  # Distribution directory
        r'(^|/)out/',  # Output directory
        r'(^|/)target/',  # Target directory
        r'(^|/)\.idea/',  # IntelliJ IDEA directory
        r'(^|/)\.vscode/',  # VS Code directory
        r'(^|/)\.gradle/',  # Gradle directory
    ]

    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Check if the file should be ignored
    for pattern in ignore_patterns:
        if re.search(pattern, file_path):
            return False

    # Check if it's a directory
    if os.path.isdir(file_path):
        return False

    # Check if the file has a code extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in code_extensions


def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is a binary file.

    Args:
        file_path: Path to the file

    Returns:
        True if it's a binary file, False otherwise
    """
    # List of common binary file extensions
    binary_extensions = [
        '.exe', '.dll', '.so', '.a', '.o', '.obj', '.bin', '.dat', '.db',
        '.class', '.jar', '.war', '.ear', '.pyc', '.pyo', '.pyd',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.svg',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv'
    ]

    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Check if the file has a binary extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in binary_extensions


def simplify_path(file_path: str) -> str:
    """
    Simplify a file path for display purposes.

    Args:
        file_path: Path to simplify

    Returns:
        Simplified path
    """
    # Get the base name of the file
    base_name = os.path.basename(file_path)

    # Get the directory name
    dir_name = os.path.dirname(file_path)

    # If the directory is deep, simplify it
    if dir_name.count(os.sep) > 2:
        parts = dir_name.split(os.sep)
        if len(parts) > 3:
            simplified_dir = os.path.join(parts[0], '...', parts[-1])
        else:
            simplified_dir = dir_name
    else:
        simplified_dir = dir_name

    # Combine simplified directory and base name
    if simplified_dir:
        return os.path.join(simplified_dir, base_name)
    else:
        return base_name


def format_timestamp(timestamp: int) -> str:
    """
    Format a Unix timestamp as a human-readable date.

    Args:
        timestamp: Unix timestamp

    Returns:
        Formatted date string
    """
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'Unknown'


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + '...'


def parse_bug_keywords(keyword_str: Optional[str] = None) -> List[str]:
    """
    Parse bug keywords from a comma-separated string.

    Args:
        keyword_str: Comma-separated keywords, or None for defaults

    Returns:
        List of keywords
    """
    default_keywords = [
        'fix', 'bug', 'defect', 'issue', 'error', 'crash', 'problem',
        'fail', 'failure', 'segfault', 'fault', 'patch', 'exception',
        'incorrect', 'mistak', 'broke', 'npe', 'corrupt', 'leak',
        'race', 'deadlock', 'hang', 'regression', 'memory', 'null',
        'resolve', 'ticket', 'jira'
    ]

    if not keyword_str:
        return default_keywords

    custom_keywords = [k.strip().lower() for k in keyword_str.split(',') if k.strip()]

    if not custom_keywords:
        return default_keywords

    return custom_keywords


def optimize_cache_size(
        repo_path: str,
        output_prefix: str,
        cache_sizes: Optional[List[float]] = None,
        policy: str = "BUG",
        lookback_commits: Optional[int] = None,
        window_ratio: float = 0.25,
        cache_seeding: bool = True,
        parallel: bool = True
) -> Dict[float, float]:
    """
    Run FixCache with different cache sizes to find the optimal size.

    Args:
        repo_path: Path to git repository
        output_prefix: Prefix for output files
        cache_sizes: List of cache sizes to test (as fraction of total files)
        policy: Cache replacement policy
        lookback_commits: Number of recent commits to analyze
        window_ratio: Ratio of commits to use for training window
        cache_seeding: Whether to seed cache with most bug-prone files
        parallel: Whether to run optimization in parallel

    Returns:
        Dictionary mapping cache sizes to hit rates
    """
    # Lazy import to avoid circular imports
    from fixcache.algorithm import FixCache
    from fixcache.visualization import plot_cache_optimization

    # Default cache sizes if not provided
    if cache_sizes is None:
        cache_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]

    logger.info(f"Running cache size optimization for {len(cache_sizes)} sizes")

    results = {}

    # Define function to run a single FixCache trial
    def run_single_trial(size: float) -> Tuple[float, float]:
        try:
            logger.info(f"Testing cache size: {size * 100:.1f}%")

            # Create FixCache instance
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=size,
                policy=policy,
                lookback_commits=lookback_commits,
                window_ratio=window_ratio,
                cache_seeding=cache_seeding
            )

            # Run repository analysis (only once per process)
            if not fix_cache.repo_analyzer.is_analyzed:
                success = fix_cache.analyze_repository()
                if not success:
                    logger.error(f"Repository analysis failed for size {size}")
                    return size, 0.0

            # Run prediction
            hit_rate = fix_cache.predict()

            # Save detailed results
            output_file = f"{output_prefix}_{int(size * 100)}.json"
            fix_cache.save_results(output_file)

            return size, hit_rate

        except Exception as e:
            logger.error(f"Error in cache size {size}: {str(e)}")
            return size, 0.0

    # Run trials (parallel or sequential)
    if parallel and len(cache_sizes) > 1 and multiprocessing.cpu_count() > 1:
        # Use process pool for parallelization
        with ProcessPoolExecutor(max_workers=min(len(cache_sizes), multiprocessing.cpu_count())) as executor:
            # Submit all trials
            future_to_size = {executor.submit(run_single_trial, size): size for size in cache_sizes}

            # Collect results as they complete
            for future in as_completed(future_to_size):
                size, hit_rate = future.result()
                results[size] = hit_rate
                logger.info(f"Cache size {size * 100:.1f}% completed: hit rate {hit_rate:.2f}%")
    else:
        # Run sequentially
        for size in cache_sizes:
            size, hit_rate = run_single_trial(size)
            results[size] = hit_rate
            logger.info(f"Cache size {size * 100:.1f}% completed: hit rate {hit_rate:.2f}%")

    # Create visualization
    visualization_file = f"{output_prefix}_chart.png"
    try:
        from fixcache.visualization import plot_cache_optimization
        plot_cache_optimization(results, visualization_file)
        logger.info(f"Optimization visualization saved to {visualization_file}")
    except Exception as e:
        logger.error(f"Error creating optimization visualization: {str(e)}")

    # Find optimal cache size
    if results:
        optimal_size = max(results.items(), key=lambda x: x[1])[0]
        logger.info(f"Optimal cache size: {optimal_size * 100:.1f}% (hit rate: {results[optimal_size]:.2f}%)")

    return results


def compare_repositories(
        repo_paths: List[str],
        output_file: str,
        cache_size: float = 0.2,
        policy: str = "BUG",
        lookback_commits: Optional[int] = None,
        visualization: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple repositories using FixCache.

    Args:
        repo_paths: List of repository paths
        output_file: Path to save results
        cache_size: Cache size as fraction of total files
        policy: Cache replacement policy
        lookback_commits: Number of recent commits to analyze
        visualization: Whether to generate visualization

    Returns:
        Dictionary mapping repository paths to their results
    """
    # Lazy import to avoid circular imports
    from fixcache.algorithm import FixCache

    results = {}

    logger.info(f"Comparing {len(repo_paths)} repositories")

    # Process each repository
    for repo_path in repo_paths:
        logger.info(f"Processing repository: {repo_path}")

        try:
            # Create FixCache instance
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=cache_size,
                policy=policy,
                lookback_commits=lookback_commits
            )

            # Run repository analysis
            if not fix_cache.analyze_repository():
                logger.error(f"Repository analysis failed for {repo_path}")
                continue

            # Run prediction
            hit_rate = fix_cache.predict()

            # Store results
            results[repo_path] = fix_cache.results

            logger.info(f"Repository {repo_path} completed: hit rate {hit_rate:.2f}%")

        except Exception as e:
            logger.error(f"Error processing repository {repo_path}: {str(e)}")

    # Save results to file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Comparison results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")

    # Create visualization
    if visualization and len(results) >= 2:
        visualization_file = os.path.splitext(output_file)[0] + '_chart.png'
        try:
            from fixcache.visualization import plot_repository_comparison
            plot_repository_comparison(results, visualization_file)
            logger.info(f"Comparison visualization saved to {visualization_file}")
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {str(e)}")

    return results


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_file: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration dictionary
        config_file: Path to configuration file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False


def get_file_type_description(file_path: str) -> str:
    """
    Get a description of a file type based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Description of the file type
    """
    # File type descriptions
    type_descriptions = {
        '.py': 'Python source code',
        '.java': 'Java source code',
        '.c': 'C source code',
        '.cpp': 'C++ source code',
        '.h': 'C/C++ header file',
        '.hpp': 'C++ header file',
        '.cs': 'C# source code',
        '.js': 'JavaScript source code',
        '.ts': 'TypeScript source code',
        '.go': 'Go source code',
        '.rb': 'Ruby source code',
        '.php': 'PHP source code',
        '.swift': 'Swift source code',
        '.kt': 'Kotlin source code',
        '.scala': 'Scala source code',
        '.rs': 'Rust source code',
        '.sh': 'Shell script',
        '.xml': 'XML file',
        '.gradle': 'Gradle build file',
        '.properties': 'Properties file',
        '.json': 'JSON file',
        '.html': 'HTML file',
        '.css': 'CSS file',
        '.md': 'Markdown file',
        '.yml': 'YAML file',
        '.yaml': 'YAML file',
        '.sql': 'SQL script',
        '.txt': 'Text file',
    }

    # Get file extension
    _, ext = os.path.splitext(file_path)

    # Return description or default
    return type_descriptions.get(ext.lower(), f'{ext[1:].upper()} file' if ext else 'Unknown file type')


def calculate_relative_risk(
        file_bugs: Dict[str, int],
        avg_bugs: float
) -> Dict[str, float]:
    """
    Calculate relative risk of bugs for each file.

    Args:
        file_bugs: Dictionary mapping files to bug counts
        avg_bugs: Average bug count per file

    Returns:
        Dictionary mapping files to relative risk scores
    """
    relative_risk = {}

    for file_path, bug_count in file_bugs.items():
        # Avoid division by zero
        if avg_bugs > 0:
            risk = bug_count / avg_bugs
        else:
            risk = bug_count

        relative_risk[file_path] = risk

    return relative_risk


def get_file_statistics(file_path: str) -> Dict[str, Any]:
    """
    Get basic statistics for a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file statistics
    """
    stats = {}

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {'error': 'File does not exist'}

        # Get file size
        stats['size'] = os.path.getsize(file_path)

        # Get file modification time
        stats['modified'] = os.path.getmtime(file_path)

        # Get file creation time
        stats['created'] = os.path.getctime(file_path)

        # Get file extension
        _, ext = os.path.splitext(file_path)
        stats['extension'] = ext

        # Get file type description
        stats['type'] = get_file_type_description(file_path)

        # Count lines of code if it's a text file
        if is_code_file(file_path) and not is_binary_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    stats['lines'] = content.count('\n') + 1
                    stats['code_lines'] = len([line for line in content.split('\n')
                                               if line.strip() and not line.strip().startswith('#')])
            except:
                stats['lines'] = 'unknown'
                stats['code_lines'] = 'unknown'

        return stats

    except Exception as e:
        return {'error': str(e)}


def merge_results(
        results_files: List[str],
        output_file: str
) -> Dict[str, Any]:
    """
    Merge multiple results files into one.

    Args:
        results_files: List of results file paths
        output_file: Path to save merged results

    Returns:
        Merged results dictionary
    """
    merged_results = {
        'merged_from': results_files,
        'timestamp': datetime.datetime.now().isoformat(),
        'repositories': {},
    }

    try:
        # Load and merge all results
        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                repo_path = data.get('repo_path', os.path.basename(file_path))
                merged_results['repositories'][repo_path] = data

            except Exception as e:
                logger.error(f"Error loading results from {file_path}: {str(e)}")

        # Save merged results
        with open(output_file, 'w') as f:
            json.dump(merged_results, f, indent=2)

        return merged_results

    except Exception as e:
        logger.error(f"Error merging results: {str(e)}")
        return merged_results


def extract_commit_statistics(
        commits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract statistics from a list of commits.

    Args:
        commits: List of commit dictionaries

    Returns:
        Dictionary with commit statistics
    """
    if not commits:
        return {
            'count': 0,
            'authors': [],
            'time_range': [0, 0],
            'files_changed': set(),
        }

    # Get unique authors
    authors = set(commit['author'] for commit in commits)

    # Get time range
    timestamps = [commit['timestamp'] for commit in commits]
    time_range = [min(timestamps), max(timestamps)]

    # Get files changed
    files_changed = set()
    for commit in commits:
        files_changed.update(commit.get('files_changed', []))

    # Calculate commit frequency
    if len(timestamps) > 1:
        time_span = (time_range[1] - time_range[0]) / (60 * 60 * 24)  # Convert to days
        frequency = len(commits) / max(1, time_span)
    else:
        frequency = 0

    return {
        'count': len(commits),
        'authors': list(authors),
        'author_count': len(authors),
        'time_range': time_range,
        'files_changed': list(files_changed),
        'file_count': len(files_changed),
        'frequency': frequency,  # Commits per day
    }