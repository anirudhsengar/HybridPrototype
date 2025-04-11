#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repository Analysis Module for FixCache

This module handles Git repository analysis, including commit history extraction,
bug fix identification, and file change tracking.

Author: anirudhsengar
"""

import os
import re
import subprocess
import logging
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from collections import defaultdict
import time
import datetime
from collections import Counter

# Setup logging
logger = logging.getLogger(__name__)


class RepositoryAnalyzer:
    """
    Handles Git repository analysis for the FixCache algorithm.

    This class extracts commit history, identifies bug-fixing commits,
    and tracks file changes to support the FixCache prediction algorithm.
    """

    def __init__(
            self,
            repo_path: str,
            bug_keywords: Optional[List[str]] = None,
            lookback_commits: Optional[int] = None,
            encoding: str = 'utf-8',
            fallback_encoding: str = 'latin-1'
    ):
        """
        Initialize the repository analyzer.

        Args:
            repo_path: Path to the git repository
            bug_keywords: List of keywords to identify bug-fixing commits
            lookback_commits: Number of recent commits to analyze (None for all)
            encoding: Primary encoding to use for git output
            fallback_encoding: Fallback encoding if primary fails
        """
        self.repo_path = os.path.abspath(repo_path)
        self.lookback_commits = lookback_commits
        self.encoding = encoding
        self.fallback_encoding = fallback_encoding

        # Enhanced bug keywords with more comprehensive patterns
        self.bug_keywords = bug_keywords or [
            'fix', 'bug', 'defect', 'issue', 'error', 'crash', 'problem',
            'fail', 'failure', 'segfault', 'fault', 'patch', 'exception',
            'incorrect', 'mistak', 'broke', 'npe', 'corrupt', 'leak',
            'race', 'deadlock', 'hang', 'regression', 'memory', 'null',
            'resolve', 'ticket', 'jira'
        ]

        # Repository data
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        self.commit_history: List[Dict[str, Any]] = []
        self.bug_fixes: List[Dict[str, Any]] = []
        self.total_files: int = 0
        self.is_analyzed: bool = False
        self.error_messages: List[str] = []

        # Patterns for bug detection
        self._compile_bug_patterns()

        logger.info(f"Initialized repository analyzer for {repo_path}")

    def _compile_bug_patterns(self) -> None:
        """Compile regex patterns for bug detection."""
        # Bug keyword pattern
        self.bug_pattern = re.compile(
            r'\b(' + '|'.join(self.bug_keywords) + r')\b',
            re.IGNORECASE
        )

        # Issue ID patterns (e.g., #123, ISSUE-456, JIRA-789)
        self.issue_pattern = re.compile(
            r'(#\d+|[A-Z]+-\d+|issue\s+\d+|bug\s+\d+)',
            re.IGNORECASE
        )

        # Short commit with file reference pattern
        self.file_ref_pattern = re.compile(
            r'(\.\w+)\b',  # Matches file extensions
            re.IGNORECASE
        )

    def analyze(self) -> bool:
        """
        Analyze the git repository to extract commit history and identify bug fixes.

        Returns:
            True if analysis was successful, False otherwise
        """
        logger.info(f"Analyzing repository: {self.repo_path}")
        start_time = time.time()

        try:
            # Change to repository directory
            original_dir = os.getcwd()
            os.chdir(self.repo_path)

            # Get list of all files in the repository
            self._get_all_files()

            # Get all commits
            self._get_all_commits()

            # Identify bug-fixing commits based on commit messages
            self._identify_bug_fixes()

            # Process each bug fix commit to extract changed files
            self._process_bug_fixes()

            # Mark repository as analyzed
            self.is_analyzed = True

            # Return to original directory
            os.chdir(original_dir)

            elapsed_time = time.time() - start_time
            logger.info(f"Repository analysis completed in {elapsed_time:.2f} seconds")
            logger.info(f"Total files: {self.total_files}")
            logger.info(f"Total commits: {len(self.commit_history)}")
            logger.info(f"Bug-fixing commits: {len(self.bug_fixes)}")

            return True

        except Exception as e:
            if original_dir != os.getcwd():
                try:
                    os.chdir(original_dir)
                except:
                    pass

            error_msg = f"Error during repository analysis: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)
            return False

    def _get_all_files(self) -> None:
        """Get all code files in the repository and initialize file stats."""
        try:
            # Use git to get all tracked files
            cmd = ["git", "ls-files"]
            output = self._run_git_command(cmd)

            if not output.strip():
                raise ValueError("No files found in repository. Is this a valid Git repository?")

            all_files = output.strip().split('\n')

            # Filter out empty entries and non-code files
            code_files = [f for f in all_files if self._is_code_file(f)]
            self.total_files = len(code_files)

            if self.total_files == 0:
                logger.warning("No code files found in repository")

            # Initialize file stats for each file
            for file_path in code_files:
                self.file_stats[file_path] = {
                    'bug_fixes': 0,
                    'last_bug_fix': None,
                    'commits': 0,
                    'authors': set(),
                    'created_in': None,
                    'extensions': os.path.splitext(file_path)[1]
                }

            logger.info(f"Found {self.total_files} code files in repository")

        except Exception as e:
            logger.error(f"Error getting files from repository: {str(e)}")
            raise

    def _is_code_file(self, file_path: str) -> bool:
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

        # Check if the file exists
        if not os.path.exists(file_path):
            return False

        # Check if the file has a code extension
        _, ext = os.path.splitext(file_path)
        return ext.lower() in code_extensions

    def _get_all_commits(self) -> None:
        """Get all commits from the repository."""
        try:
            # Determine commit limit argument
            limit_arg = []
            if self.lookback_commits:
                limit_arg = [f"-{self.lookback_commits}"]

            # Use git to get all commits with metadata
            cmd = ["git", "log"] + limit_arg + ["--pretty=format:%H|%an|%at|%s"]
            output = self._run_git_command(cmd)

            if not output.strip():
                logger.warning("No commits found in repository")
                return

            commits = output.strip().split('\n')

            # Parse commit information
            for commit in commits:
                parts = commit.split('|', 3)
                if len(parts) == 4:
                    sha, author, timestamp, message = parts
                    commit_info = {
                        'sha': sha,
                        'author': author,
                        'timestamp': int(timestamp),
                        'message': message,
                        'is_bug_fix': False,
                        'files_changed': []
                    }
                    self.commit_history.append(commit_info)

            logger.info(f"Retrieved {len(self.commit_history)} commits from repository")

        except Exception as e:
            logger.error(f"Error getting commits from repository: {str(e)}")
            raise

    def _identify_bug_fixes(self) -> None:
        """Identify bug-fixing commits based on commit messages."""
        for commit in self.commit_history:
            message = commit['message'].lower()

            # Check if message contains bug keywords
            if self.bug_pattern.search(message):
                commit['is_bug_fix'] = True
                self.bug_fixes.append(commit)
                continue

            # Check for issue references
            if self.issue_pattern.search(message):
                commit['is_bug_fix'] = True
                self.bug_fixes.append(commit)
                continue

            # Heuristic 1: short message with file references
            if len(message.split()) < 10 and self.file_ref_pattern.search(message):
                commit['is_bug_fix'] = True
                self.bug_fixes.append(commit)
                continue

            # Heuristic 2: messages with single statement
            if message.startswith('fix') or message.startswith('resolve'):
                commit['is_bug_fix'] = True
                self.bug_fixes.append(commit)
                continue

        logger.info(f"Identified {len(self.bug_fixes)} bug-fixing commits")

    def _process_bug_fixes(self) -> None:
        """Process each bug-fixing commit to extract changed files."""
        for i, commit in enumerate(self.bug_fixes):
            # Get files changed in this commit
            files_changed = self._get_files_changed(commit['sha'])
            commit['files_changed'] = files_changed

            # Update file stats for each file
            for file_path in files_changed:
                if file_path in self.file_stats:
                    self.file_stats[file_path]['bug_fixes'] += 1
                    self.file_stats[file_path]['last_bug_fix'] = commit['timestamp']
                    self.file_stats[file_path]['authors'].add(commit['author'])

            # Log progress for large repositories
            if i > 0 and i % 100 == 0:
                logger.info(f"Processed {i} of {len(self.bug_fixes)} bug-fixing commits")

    def _get_files_changed(self, commit_sha: str) -> List[str]:
        """
        Get the list of files changed in a commit.

        Args:
            commit_sha: The SHA of the commit

        Returns:
            List of file paths changed in the commit
        """
        try:
            # Use git to get files changed in this commit
            cmd = ["git", "show", "--name-only", "--pretty=format:", commit_sha]
            output = self._run_git_command(cmd)

            # Filter out empty lines and non-existing files
            files = [f for f in output.strip().split('\n') if f and os.path.exists(f)]

            return files

        except Exception as e:
            logger.error(f"Error getting files changed in commit {commit_sha}: {str(e)}")
            return []

    def _run_git_command(self, cmd: List[str]) -> str:
        """
        Run a git command with robust error and encoding handling.

        Args:
            cmd: Command to run as a list of strings

        Returns:
            Command output as a string
        """
        try:
            # First try with primary encoding
            return self._run_command_with_encoding(cmd, self.encoding)
        except UnicodeDecodeError:
            # If that fails, try with fallback encoding
            logger.warning(f"Failed to decode git output with {self.encoding}, "
                           f"falling back to {self.fallback_encoding}")
            try:
                return self._run_command_with_encoding(cmd, self.fallback_encoding)
            except UnicodeDecodeError:
                # Last resort: ignore errors
                logger.warning(f"Failed to decode git output with fallback encoding, "
                               "using 'replace' error handling")
                return self._run_command_with_encoding(cmd, self.encoding, 'replace')
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}")
            if e.output:
                logger.error(f"Output: {e.output}")
            raise

    def _run_command_with_encoding(
            self,
            cmd: List[str],
            encoding: str,
            errors: str = 'strict'
    ) -> str:
        """
        Run a command with specific encoding and error handling.

        Args:
            cmd: Command to run
            encoding: Character encoding to use
            errors: Error handling strategy ('strict', 'replace', etc.)

        Returns:
            Command output as string
        """
        try:
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                universal_newlines=False  # Get bytes output
            )
            return output.decode(encoding, errors=errors)
        except subprocess.CalledProcessError as e:
            if hasattr(e, 'output') and e.output:
                # Try to decode error output
                try:
                    error_output = e.output.decode(encoding, errors='replace')
                    logger.error(f"Command error output: {error_output}")
                except:
                    pass
            raise

    def get_file_change_frequency(self) -> Dict[str, float]:
        """
        Calculate change frequency for each file.

        Returns:
            Dictionary mapping file paths to change frequency
        """
        frequencies = {}

        # Get the earliest and latest commit timestamps
        if not self.commit_history:
            return {}

        earliest = min(commit['timestamp'] for commit in self.commit_history)
        latest = max(commit['timestamp'] for commit in self.commit_history)
        time_span = max(1, (latest - earliest) / (60 * 60 * 24))  # Convert to days

        # Calculate number of changes per day for each file
        for file_path, stats in self.file_stats.items():
            frequency = stats['bug_fixes'] / time_span
            frequencies[file_path] = frequency

        return frequencies

    def get_file_complexity(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate complexity metrics for files.

        Returns:
            Dictionary mapping file paths to complexity metrics
        """
        complexity_metrics = {}

        for file_path in self.file_stats:
            if not os.path.exists(file_path):
                continue

            try:
                # Get file size
                file_size = os.path.getsize(file_path)

                # Get line count (simple complexity metric)
                with open(file_path, 'r', encoding=self.encoding, errors='replace') as f:
                    line_count = sum(1 for _ in f)

                complexity_metrics[file_path] = {
                    'size_bytes': file_size,
                    'line_count': line_count,
                    'size_category': self._categorize_file_size(file_size),
                    'bug_density': self.file_stats[file_path]['bug_fixes'] / max(1, line_count)
                }

            except Exception as e:
                logger.warning(f"Error calculating complexity for {file_path}: {str(e)}")

        return complexity_metrics

    def _categorize_file_size(self, size_bytes: int) -> str:
        """
        Categorize file size into small, medium, large.

        Args:
            size_bytes: File size in bytes

        Returns:
            Category string
        """
        if size_bytes < 1000:  # 1 KB
            return "small"
        elif size_bytes < 10000:  # 10 KB
            return "medium"
        else:
            return "large"

    def get_file_ownership(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate file ownership metrics.

        Returns:
            Dictionary mapping file paths to ownership metrics
        """
        ownership = {}

        for file_path, stats in self.file_stats.items():
            authors = stats['authors']
            ownership[file_path] = {
                'author_count': len(authors),
                'has_multiple_authors': len(authors) > 1,
            }

        return ownership

    def get_repository_summary(self) -> Dict[str, Any]:
        """
        Get a summary of repository metrics.

        Returns:
            Dictionary with repository summary
        """
        if not self.is_analyzed:
            logger.warning("Repository has not been analyzed. Run analyze() first.")
            return {}

        # Get various statistics
        file_extensions = Counter([os.path.splitext(f)[1] for f in self.file_stats])
        top_extensions = file_extensions.most_common(5)

        # Count commits per author
        author_commits = Counter()
        for commit in self.commit_history:
            author_commits[commit['author']] += 1
        top_authors = author_commits.most_common(5)

        # Calculate average bug fixes per file
        avg_bug_fixes = sum(stats['bug_fixes'] for stats in self.file_stats.values()) / max(1, len(self.file_stats))

        return {
            'name': os.path.basename(self.repo_path),
            'total_files': self.total_files,
            'total_commits': len(self.commit_history),
            'bug_fixing_commits': len(self.bug_fixes),
            'unique_authors': len(set(commit['author'] for commit in self.commit_history)),
            'top_file_extensions': top_extensions,
            'top_authors': top_authors,
            'bug_fix_ratio': len(self.bug_fixes) / max(1, len(self.commit_history)),
            'avg_bug_fixes_per_file': avg_bug_fixes,
            'analyzed_at': datetime.datetime.now().isoformat(),
        }

    def get_bug_fix_distribution(self) -> Dict[str, int]:
        """
        Get distribution of bug fixes across different file types.

        Returns:
            Dictionary mapping file extensions to bug fix counts
        """
        extension_bug_counts = defaultdict(int)

        for file_path, stats in self.file_stats.items():
            ext = os.path.splitext(file_path)[1]
            extension_bug_counts[ext] += stats['bug_fixes']

        return dict(extension_bug_counts)

    def get_temporal_bug_distribution(self) -> Dict[str, List[int]]:
        """
        Get temporal distribution of bug fixes.

        Returns:
            Dictionary with time periods and bug fix counts
        """
        if not self.bug_fixes:
            return {'labels': [], 'counts': []}

        # Sort bug fixes by timestamp
        sorted_fixes = sorted(self.bug_fixes, key=lambda x: x['timestamp'])

        # Get time range
        start_time = sorted_fixes[0]['timestamp']
        end_time = sorted_fixes[-1]['timestamp']
        time_span = end_time - start_time

        # Determine appropriate time division (day, week, month)
        if time_span < 60 * 60 * 24 * 30:  # Less than a month
            # Group by day
            period = 60 * 60 * 24
            format_str = "%Y-%m-%d"
            period_name = "day"
        elif time_span < 60 * 60 * 24 * 30 * 6:  # Less than 6 months
            # Group by week
            period = 60 * 60 * 24 * 7
            format_str = "%Y-W%W"
            period_name = "week"
        else:
            # Group by month
            period = 60 * 60 * 24 * 30
            format_str = "%Y-%m"
            period_name = "month"

        # Create time buckets
        num_periods = max(1, int(time_span / period)) + 1
        period_labels = []
        period_counts = [0] * num_periods

        for i in range(num_periods):
            period_time = start_time + i * period
            label = datetime.datetime.fromtimestamp(period_time).strftime(format_str)
            period_labels.append(label)

        # Count bugs in each period
        for commit in self.bug_fixes:
            period_index = min(num_periods - 1, int((commit['timestamp'] - start_time) / period))
            period_counts[period_index] += 1

        return {
            'period_type': period_name,
            'labels': period_labels,
            'counts': period_counts
        }