#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Module for FixCache

This module defines configuration parameters, constants, and defaults
used throughout the FixCache bug prediction tool.

Author: anirudhsengar
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Set

# Setup logging
logger = logging.getLogger(__name__)

# Version information
VERSION = "0.1.0"
TOOL_NAME = "FixCachePrototype"
TOOL_URL = "https://github.com/anirudhsengar/FixCachePrototype"

# Default configuration values
DEFAULT_CONFIG = {
    # Algorithm parameters
    "cache_size": 0.2,  # Default cache size (20% of files)
    "policy": "BUG",  # Default cache replacement policy
    "window_ratio": 0.25,  # Default training window ratio
    "cache_seeding": True,  # Whether to seed cache with most bug-prone files
    "min_file_count": 10,  # Minimum files required for analysis

    # Repository analysis
    "bug_keywords": [
        "fix", "bug", "defect", "issue", "error", "crash", "problem",
        "fail", "failure", "segfault", "fault", "patch", "exception",
        "incorrect", "mistak", "broke", "npe", "corrupt", "leak",
        "race", "deadlock", "hang", "regression", "memory", "null",
        "resolve", "ticket", "jira"
    ],
    "exclude_patterns": [
        r"(^|/)\.git/",
        r"(^|/)node_modules/",
        r"(^|/)build/",
        r"(^|/)dist/",
        r"(^|/)out/",
        r"(^|/)target/",
        r"(^|/)\.idea/",
        r"(^|/)\.vscode/",
        r"(^|/)\.gradle/"
    ],
    "lookback_commits": None,  # Number of recent commits to analyze (None = all)
    "primary_encoding": "utf-8",  # Primary encoding for repository files
    "fallback_encoding": "latin-1",  # Fallback encoding if primary fails

    # Visualization
    "visualization_enabled": True,  # Whether to generate visualizations
    "visualization_dpi": 150,  # DPI for visualizations
    "theme_colors": {
        "primary": "#1f77b4",  # Blue
        "secondary": "#ff7f0e",  # Orange
        "tertiary": "#2ca02c",  # Green
        "quaternary": "#d62728",  # Red
        "background": "#f8f9fa",
        "text": "#333333"
    },

    # Paths and file locations
    "config_path": "~/.fixcache",
    "output_dir": ".",
    "log_level": "INFO"
}

# Valid cache replacement policies
VALID_POLICIES = ["BUG", "FIFO", "LRU"]

# File type categories
FILE_CATEGORIES = {
    "source": [".py", ".java", ".c", ".cpp", ".cs", ".js", ".ts", ".go", ".rb", ".php"],
    "header": [".h", ".hpp", ".hxx"],
    "script": [".sh", ".bat", ".ps1", ".bash"],
    "markup": [".html", ".xml", ".md", ".rst", ".json", ".yaml", ".yml"],
    "config": [".properties", ".conf", ".config", ".ini", ".gradle", ".toml"],
    "data": [".csv", ".txt", ".sql", ".log"]
}

# File Type Descriptions
FILE_TYPE_DESCRIPTIONS = {
    ".py": "Python source code",
    ".java": "Java source code",
    ".c": "C source code",
    ".cpp": "C++ source code",
    ".h": "C/C++ header file",
    ".hpp": "C++ header file",
    ".cs": "C# source code",
    ".js": "JavaScript source code",
    ".ts": "TypeScript source code",
    ".go": "Go source code",
    ".rb": "Ruby source code",
    ".php": "PHP source code",
    ".swift": "Swift source code",
    ".kt": "Kotlin source code",
    ".scala": "Scala source code",
    ".rs": "Rust source code",
    ".sh": "Shell script",
    ".xml": "XML file",
    ".gradle": "Gradle build file",
    ".properties": "Properties file",
    ".json": "JSON file",
    ".html": "HTML file",
    ".css": "CSS file",
    ".md": "Markdown file",
    ".yml": "YAML file",
    ".yaml": "YAML file",
    ".sql": "SQL script",
    ".txt": "Text file",
}

# Hit rate interpretation thresholds
HIT_RATE_THRESHOLDS = {
    "excellent": 80.0,
    "very_good": 60.0,
    "good": 40.0,
    "fair": 20.0,
    "poor": 0.0
}


class Config:
    """
    Configuration class for the FixCache tool.

    This class manages configuration settings, loading/saving config files,
    and provides access to configuration values throughout the application.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration with default values and optionally load from file.

        Args:
            config_file: Path to configuration file to load (optional)
        """
        # Start with default configuration
        self.config = DEFAULT_CONFIG.copy()

        # Creation timestamp
        self.created_at = datetime.datetime.now().isoformat()

        # Load configuration if provided
        if config_file:
            self.load(config_file)

    def load(self, config_file: str) -> bool:
        """
        Load configuration from a JSON file.

        Args:
            config_file: Path to configuration file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(os.path.expanduser(config_file), 'r') as f:
                loaded_config = json.load(f)

            # Update configuration with loaded values
            self.config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_file}")
            return True

        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_file}")
            return False

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_file}")
            return False

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False

    def save(self, config_file: str) -> bool:
        """
        Save configuration to a JSON file.

        Args:
            config_file: Path to save configuration

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            config_dir = os.path.dirname(os.path.expanduser(config_file))
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            with open(os.path.expanduser(config_file), 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Saved configuration to {config_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        self.config.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate cache size
        cache_size = self.config.get("cache_size")
        if not (isinstance(cache_size, (int, float)) and 0 < cache_size <= 1):
            errors.append(f"Invalid cache size: {cache_size}. Must be between 0 and 1.")

        # Validate policy
        policy = self.config.get("policy")
        if policy not in VALID_POLICIES:
            errors.append(f"Invalid policy: {policy}. Must be one of {VALID_POLICIES}.")

        # Validate window ratio
        window_ratio = self.config.get("window_ratio")
        if not (isinstance(window_ratio, (int, float)) and 0 < window_ratio <= 1):
            errors.append(f"Invalid window ratio: {window_ratio}. Must be between 0 and 1.")

        return errors

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary-like access.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary-like access.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the configuration.

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        return key in self.config


# Helper functions for working with configuration

def get_config_path() -> str:
    """
    Get the path to the user's configuration file.

    Returns:
        Path to configuration file
    """
    # Check for environment variable override
    env_config = os.environ.get("FIXCACHE_CONFIG")
    if env_config:
        return env_config

    # Get default config path
    default_path = os.path.expanduser(DEFAULT_CONFIG["config_path"])

    # Create directory if it doesn't exist
    config_dir = os.path.dirname(default_path)
    if config_dir and not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir)
        except:
            pass

    return os.path.join(default_path, "config.json")


def load_global_config() -> Config:
    """
    Load the global configuration.

    Returns:
        Config object
    """
    config_path = get_config_path()
    config = Config(config_path if os.path.exists(config_path) else None)
    return config


def get_cache_policy_description(policy: str) -> str:
    """
    Get a description of a cache replacement policy.

    Args:
        policy: Cache policy name

    Returns:
        Description of the policy
    """
    descriptions = {
        "BUG": "Bug-based policy: Evict file with fewest bug fixes",
        "FIFO": "First-In-First-Out: Evict oldest file in cache",
        "LRU": "Least Recently Used: Evict least recently accessed file"
    }

    return descriptions.get(policy, f"Unknown policy: {policy}")


def get_default_bug_keywords() -> List[str]:
    """
    Get the default list of bug keywords.

    Returns:
        List of bug-related keywords
    """
    return DEFAULT_CONFIG["bug_keywords"]


def get_tool_info() -> Dict[str, str]:
    """
    Get information about the tool.

    Returns:
        Dictionary with tool information
    """
    return {
        "name": TOOL_NAME,
        "version": VERSION,
        "url": TOOL_URL,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "author": "anirudhsengar",
        "license": "Eclipse Public License 2.0"
    }


# Create a global config instance
global_config = load_global_config()