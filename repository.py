#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Repository Interface for Hybrid Bug Predictor

This module provides a unified interface for repository analysis that can be used
by both FixCache and REPD implementations, combining functionality from both approaches.

Author: anirudhsengar
"""

import os
import re
import logging
import subprocess
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Any, Optional, Union, Generator
from collections import defaultdict
import git
from git import Repo, Commit as GitCommit

logger = logging.getLogger(__name__)


class Commit:
    """Represents a commit in version control history."""

    def __init__(self,
                 hash: str,
                 author: str,
                 timestamp: int,
                 date: datetime,
                 message: str,
                 files_changed: List[str],
                 is_bug_fix: bool = False,
                 parent_hash: Optional[str] = None):
        self.hash = hash
        self.author = author
        self.timestamp = timestamp
        self.date = date
        self.message = message
        self.files_changed = files_changed
        self.is_bug_fix = is_bug_fix
        self.parent_hash = parent_hash

        # Additional fields for REPD analysis
        self.metrics = {}
        self.complexity = None
        self.change_coupling = {}


class FileChange:
    """Represents a file change in a commit."""

    def __init__(self,
                 file_path: str,
                 commit_hash: str,
                 change_type: str,  # 'A' (added), 'M' (modified), 'D' (deleted), 'R' (renamed)
                 old_path: Optional[str] = None,
                 lines_added: int = 0,
                 lines_deleted: int = 0):
        self.file_path = file_path
        self.commit_hash = commit_hash
        self.change_type = change_type
        self.old_path = old_path
        self.lines_added = lines_added
        self.lines_deleted = lines_deleted
        self.timestamp = None  # Will be set after associating with commit


class FileMetrics:
    """Stores metrics for a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.size = 0  # Size in bytes
        self.loc = 0  # Lines of code
        self.complexity = 0  # Cyclomatic complexity
        self.author_count = 0  # Number of authors who modified this file
        self.commit_count = 0  # Number of commits that modified this file
        self.bug_fix_count = 0  # Number of bug fix commits that modified this file
        self.last_modified = None  # Timestamp of last modification
        self.creation_date = None  # Timestamp of file creation
        # Additional REPD metrics
        self.coupling_scores = {}  # Files coupled with this one
        self.change_history = []  # List of changes to this file
        self.reconstruction_error = 0.0  # REPD reconstruction error


class Repository:
    """
    Unified repository analyzer that combines features needed by
    both FixCache and REPD approaches.
    """

    def __init__(
            self,
            repo_path: str,
            bug_keywords: Optional[List[str]] = None,
            lookback_commits: Optional[int] = None,
            lookback_days: Optional[int] = None,
            encoding: str = 'utf-8',
            fallback_encoding: str = 'latin-1',
            clone_if_missing: bool = True
    ):
        """
        Initialize the repository analyzer.

        Args:
            repo_path: Local path or remote URL of git repository
            bug_keywords: List of keywords to identify bug fixes in commit messages
            lookback_commits: Maximum number of commits to analyze (None = all)
            lookback_days: Maximum age of commits to analyze in days (None = all)
            encoding: Primary encoding for file reading
            fallback_encoding: Fallback encoding if primary fails
            clone_if_missing: Whether to clone the repository if the path doesn't exist
        """
        # Setup repository path
        self.is_remote = repo_path.startswith(('http://', 'https://', 'git://'))
        self.original_path = repo_path

        if self.is_remote and clone_if_missing:
            # For remote repos, clone to a temporary directory
            self.repo_path = self._clone_repository(repo_path)
        else:
            # Local repository
            self.repo_path = os.path.abspath(repo_path)

            # Check if the repository exists and is a git repo
            if not os.path.exists(os.path.join(self.repo_path, '.git')) and clone_if_missing:
                # Try to clone if it's a remote repo
                if repo_path.startswith(('https://', 'http://', 'git://')):
                    self.repo_path = self._clone_repository(repo_path)
                else:
                    raise ValueError(f"Path {repo_path} is not a valid git repository")

        # Initialize git repository
        self.repo = Repo(self.repo_path)

        # Configuration parameters
        self.lookback_commits = lookback_commits
        self.lookback_days = lookback_days
        self.encoding = encoding
        self.fallback_encoding = fallback_encoding

        # Default bug keywords
        self.bug_keywords = bug_keywords or [
            'fix', 'bug', 'defect', 'issue', 'error', 'crash', 'problem',
            'fail', 'failure', 'segfault', 'fault', 'patch', 'exception',
            'incorrect', 'mistak', 'broke', 'npe', 'corrupt', 'leak',
            'race', 'deadlock', 'hang', 'regression', 'memory', 'null',
            'resolve', 'ticket', 'jira'
        ]

        # Repository data structures
        self.commits = []  # List of Commit objects
        self.bug_fixes = []  # List of bug fix commits
        self.file_changes = defaultdict(list)  # Dict mapping file paths to lists of FileChange objects
        self.file_metrics = {}  # Dict mapping file paths to FileMetrics objects
        self.file_last_changes = {}  # Dict mapping file paths to their last change commit hash
        self.commit_map = {}  # Dict mapping commit hashes to Commit objects

        # Analysis state
        self.total_files = 0
        self.is_analyzed = False
        self.error_messages = []

        # Compile regex patterns for bug detection
        self._compile_bug_patterns()

    def _clone_repository(self, repo_url: str) -> str:
        """
        Clone a remote repository to a temporary directory.

        Args:
            repo_url: URL of the repository to clone

        Returns:
            Path to the cloned repository
        """
        # Create a temporary directory name based on the repo URL
        repo_name = repo_url.split('/')[-1].split('.')[0]
        temp_dir = os.path.join(os.getcwd(), f"temp_{repo_name}_{int(time.time())}")

        logger.info(f"Cloning {repo_url} to {temp_dir}")

        try:
            Repo.clone_from(repo_url, temp_dir)
            logger.info(f"Repository cloned successfully to {temp_dir}")
            return temp_dir
        except git.GitCommandError as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            raise ValueError(f"Failed to clone repository: {str(e)}")

    def _compile_bug_patterns(self) -> None:
        """Compile regex patterns for bug detection."""
        # Bug keyword pattern (word boundaries ensure we match whole words)
        self.bug_pattern = re.compile(
            r'\b(' + '|'.join(self.bug_keywords) + r')\b',
            re.IGNORECASE
        )

        # Issue reference patterns (e.g., #123, ISSUE-456, JIRA-789)
        self.issue_pattern = re.compile(
            r'(#\d+|[A-Z]+-\d+|issue\s+\d+|bug\s+\d+)',
            re.IGNORECASE
        )

    def analyze(self) -> bool:
        """
        Analyze the git repository to extract commit history and identify bug fixes.

        Returns:
            True if analysis was successful, False otherwise
        """
        if self.is_analyzed:
            logger.info("Repository already analyzed, skipping analysis")
            return True

        logger.info(f"Starting analysis of repository {self.repo_path}")
        start_time = time.time()

        try:
            # Set up the git command to get commits
            git_cmd = ['git', 'log', '--no-merges', '--name-status', '--date=raw']

            # Add lookback limit if specified
            if self.lookback_commits:
                git_cmd.append(f"-{self.lookback_commits}")

            # Add date limit if specified
            if self.lookback_days:
                since_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
                git_cmd.extend(['--since', since_date])

            # Run the git command
            process = subprocess.Popen(
                git_cmd,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"Git log command failed: {stderr}")
                self.error_messages.append(f"Git log error: {stderr}")
                return False

            # Process the commit log
            self._process_git_log(stdout)

            # Calculate file metrics
            self._calculate_file_metrics()

            # Mark analysis as complete
            self.is_analyzed = True
            self.total_files = len(self.file_metrics)

            logger.info(f"Repository analysis completed in {time.time() - start_time:.2f}s")
            logger.info(f"Found {len(self.commits)} commits, {len(self.bug_fixes)} bug fixes, {self.total_files} files")

            return True

        except Exception as e:
            logger.exception(f"Error analyzing repository: {str(e)}")
            self.error_messages.append(f"Analysis error: {str(e)}")
            return False

    def _process_git_log(self, git_log_output: str) -> None:
        """
        Process git log output to extract commits and file changes.

        Args:
            git_log_output: Output from git log command
        """
        lines = git_log_output.strip().split('\n')

        current_commit = None
        current_files = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # New commit starts with "commit <hash>"
            if line.startswith('commit '):
                # Save the previous commit if it exists
                if current_commit:
                    # Create a Commit object and store it
                    is_bug_fix = self._is_bug_fix(current_commit['message'])
                    commit = Commit(
                        hash=current_commit['hash'],
                        author=current_commit['author'],
                        timestamp=current_commit['timestamp'],
                        date=datetime.fromtimestamp(current_commit['timestamp']),
                        message=current_commit['message'],
                        files_changed=[file['path'] for file in current_files],
                        is_bug_fix=is_bug_fix,
                        parent_hash=current_commit.get('parent')
                    )

                    self.commits.append(commit)
                    self.commit_map[commit.hash] = commit

                    # Store bug fixes separately
                    if is_bug_fix:
                        self.bug_fixes.append(commit)

                    # Store file changes
                    for file_change in current_files:
                        change = FileChange(
                            file_path=file_change['path'],
                            commit_hash=current_commit['hash'],
                            change_type=file_change['status'],
                            old_path=file_change.get('old_path'),
                            lines_added=file_change.get('lines_added', 0),
                            lines_deleted=file_change.get('lines_deleted', 0)
                        )
                        change.timestamp = current_commit['timestamp']
                        self.file_changes[file_change['path']].append(change)
                        self.file_last_changes[file_change['path']] = current_commit['hash']

                # Start a new commit
                commit_hash = line.split(' ')[1]
                current_commit = {'hash': commit_hash, 'files': []}
                current_files = []
                i += 1

            # Author line
            elif line.startswith('Author: '):
                current_commit['author'] = line[8:].strip()
                i += 1

            # Date line
            elif line.startswith('Date: '):
                timestamp_parts = line.split(' ')
                current_commit['timestamp'] = int(timestamp_parts[-2])
                i += 1

            # Parent line (extracted for relationship tracking)
            elif line.startswith('parent '):
                current_commit['parent'] = line.split(' ')[1]
                i += 1

            # Blank line followed by commit message
            elif not line and i + 1 < len(lines):
                # Extract commit message
                message_lines = []
                i += 1
                while i < len(lines) and not (lines[i].startswith('commit ') or
                                              lines[i].startswith('A\t') or
                                              lines[i].startswith('M\t') or
                                              lines[i].startswith('D\t') or
                                              lines[i].startswith('R')):
                    if lines[i].strip():
                        message_lines.append(lines[i].strip())
                    i += 1
                current_commit['message'] = '\n'.join(message_lines).strip()

                # Process file changes
                while i < len(lines) and not lines[i].startswith('commit '):
                    file_line = lines[i].strip()
                    if not file_line:
                        i += 1
                        continue

                    # Parse file change line
                    if file_line[0] in ['A', 'M', 'D']:
                        # Simple add/modify/delete change
                        status, path = file_line.split('\t', 1)
                        current_files.append({
                            'path': path,
                            'status': status
                        })
                    elif file_line.startswith('R'):
                        # Rename - format is R<percentage>\t<old_path>\t<new_path>
                        parts = file_line.split('\t')
                        status = parts[0]
                        if len(parts) >= 3:
                            old_path, new_path = parts[1], parts[2]
                            current_files.append({
                                'path': new_path,
                                'old_path': old_path,
                                'status': 'R'
                            })
                    i += 1
            else:
                # Skip unrecognized line
                i += 1

        # Don't forget the last commit
        if current_commit:
            is_bug_fix = self._is_bug_fix(current_commit['message'])
            commit = Commit(
                hash=current_commit['hash'],
                author=current_commit['author'],
                timestamp=current_commit['timestamp'],
                date=datetime.fromtimestamp(current_commit['timestamp']),
                message=current_commit['message'],
                files_changed=[file['path'] for file in current_files],
                is_bug_fix=is_bug_fix,
                parent_hash=current_commit.get('parent')
            )

            self.commits.append(commit)
            self.commit_map[commit.hash] = commit

            if is_bug_fix:
                self.bug_fixes.append(commit)

            for file_change in current_files:
                change = FileChange(
                    file_path=file_change['path'],
                    commit_hash=current_commit['hash'],
                    change_type=file_change['status'],
                    old_path=file_change.get('old_path'),
                    lines_added=file_change.get('lines_added', 0),
                    lines_deleted=file_change.get('lines_deleted', 0)
                )
                change.timestamp = current_commit['timestamp']
                self.file_changes[file_change['path']].append(change)
                self.file_last_changes[file_change['path']] = current_commit['hash']

    def _is_bug_fix(self, commit_message: str) -> bool:
        """
        Determine if a commit message indicates a bug fix.

        Args:
            commit_message: The commit message to analyze

        Returns:
            True if the commit appears to be a bug fix, False otherwise
        """
        message_lower = commit_message.lower()

        # Check for bug keywords
        if self.bug_pattern.search(message_lower):
            return True

        # Check for issue references
        if self.issue_pattern.search(commit_message):
            return True

        return False

    def _calculate_file_metrics(self) -> None:
        """
        Calculate metrics for all files in the repository.
        """
        # Get current files in the repository
        all_files = set()
        for commit in self.commits:
            for file_path in commit.files_changed:
                all_files.add(file_path)

        # Create metrics objects for each file
        for file_path in all_files:
            if file_path not in self.file_metrics:
                self.file_metrics[file_path] = FileMetrics(file_path)

            metrics = self.file_metrics[file_path]

            # Get file changes
            file_changes = self.file_changes.get(file_path, [])

            # Calculate basic metrics
            metrics.commit_count = len(file_changes)

            # Calculate bug fix count
            metrics.bug_fix_count = sum(1 for change in file_changes
                                        if change.commit_hash in self.commit_map and
                                        self.commit_map[change.commit_hash].is_bug_fix)

            # Get creation date and last modified date
            if file_changes:
                sorted_changes = sorted(file_changes, key=lambda c: c.timestamp)
                metrics.creation_date = sorted_changes[0].timestamp
                metrics.last_modified = sorted_changes[-1].timestamp

            # Calculate LOC and file size if file exists
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    metrics.size = os.path.getsize(full_path)

                    # Try to count lines of code
                    try:
                        with open(full_path, 'r', encoding=self.encoding) as f:
                            metrics.loc = sum(1 for _ in f)
                    except UnicodeDecodeError:
                        try:
                            with open(full_path, 'r', encoding=self.fallback_encoding) as f:
                                metrics.loc = sum(1 for _ in f)
                        except:
                            # If we can't read the file, just use file size as a proxy
                            metrics.loc = metrics.size // 100  # rough approximation
                except:
                    pass

            # Get unique authors
            authors = set(self.commit_map[change.commit_hash].author
                          for change in file_changes
                          if change.commit_hash in self.commit_map)
            metrics.author_count = len(authors)

    def get_all_files(self) -> List[str]:
        """
        Get a list of all files in the repository.

        Returns:
            List of file paths
        """
        if not self.is_analyzed:
            self.analyze()

        return list(self.file_metrics.keys())

    def get_current_files(self) -> List[str]:
        """
        Get a list of files currently in the repository (not deleted).

        Returns:
            List of file paths for files that currently exist
        """
        if not self.is_analyzed:
            self.analyze()

        current_files = []
        for file_path in self.file_metrics:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                current_files.append(file_path)

        return current_files

    def get_buggy_files(self) -> List[str]:
        """
        Get a list of files that have been involved in bug fixes.

        Returns:
            List of file paths
        """
        if not self.is_analyzed:
            self.analyze()

        buggy_files = set()
        for commit in self.bug_fixes:
            for file_path in commit.files_changed:
                buggy_files.add(file_path)

        return list(buggy_files)

    def get_metrics_for_file(self, file_path: str) -> Optional[FileMetrics]:
        """
        Get metrics for a specific file.

        Args:
            file_path: Path to the file (relative to repo root)

        Returns:
            FileMetrics object or None if file not found
        """
        if not self.is_analyzed:
            self.analyze()

        return self.file_metrics.get(file_path)

    def get_commits_for_file(self, file_path: str) -> List[Commit]:
        """
        Get all commits that modified a specific file.

        Args:
            file_path: Path to the file (relative to repo root)

        Returns:
            List of Commit objects
        """
        if not self.is_analyzed:
            self.analyze()

        file_changes = self.file_changes.get(file_path, [])
        commits = []

        for change in file_changes:
            if change.commit_hash in self.commit_map:
                commits.append(self.commit_map[change.commit_hash])

        return commits

    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the repository.

        Returns:
            Dictionary with repository statistics
        """
        if not self.is_analyzed:
            self.analyze()

        return {
            "total_commits": len(self.commits),
            "bug_fixes": len(self.bug_fixes),
            "total_files": len(self.file_metrics),
            "current_files": len(self.get_current_files()),
            "buggy_files": len(self.get_buggy_files()),
            "all_files": list(self.file_metrics.keys()),
            "authors": len(set(commit.author for commit in self.commits))
        }

    def cleanup(self) -> None:
        """
        Clean up temporary resources, such as cloned repositories.
        """
        if self.is_remote and os.path.exists(self.repo_path):
            logger.info(f"Cleaning up temporary repository at {self.repo_path}")
            shutil.rmtree(self.repo_path, ignore_errors=True)

    def __del__(self) -> None:
        """
        Destructor to ensure cleanup of temporary resources.
        """
        try:
            self.cleanup()
        except:
            pass