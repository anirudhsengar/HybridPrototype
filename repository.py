#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Repository Interface for Hybrid Bug Predictor

This module provides a unified interface for repository analysis that can be used
by both FixCache and REPD implementations.

Author: anirudhsengar
"""

import os
import re
import logging
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Any, Optional, Union

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
                 is_bug_fix: bool = False):
        self.hash = hash
        self.author = author
        self.timestamp = timestamp
        self.date = date
        self.message = message
        self.files_changed = files_changed
        self.is_bug_fix = is_bug_fix


class Repository:
    """
    Unified repository analyzer that combines features needed by
    both FixCache and REPD.
    """

    def __init__(
            self,
            repo_path: str,
            bug_keywords: Optional[List[str]] = None,
            lookback_commits: Optional[int] = None,
            encoding: str = 'utf-8',
            fallback_encoding: str = 'latin-1'
    ):
        """Initialize the repository analyzer."""
        self.repo_path = os.path.abspath(repo_path)
        self.lookback_commits = lookback_commits
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

        # Repository data
        self.file_stats = {}
        self.commit_history = []
        self.bug_fixes = []
        self.total_files = 0
        self.is_analyzed = False
        self.error_messages = []

        # Patterns for bug detection
        self._compile_bug_patterns()

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

        # File reference pattern
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
        # Implement combined analysis logic here, similar to both prototypes
        # ...

    # Other methods needed by both FixCache and REPD
    # ...