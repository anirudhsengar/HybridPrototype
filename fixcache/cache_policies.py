#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Replacement Policies

This module implements different cache replacement policies for the FixCache algorithm.

Author: anirudhsengar
"""


def get_cache_policy(policy_name):
    """
    Return the cache replacement policy function based on the policy name.

    Args:
        policy_name (str): The name of the policy ("BUG", "FIFO", or "LRU")

    Returns:
        function: The policy function

    Raises:
        ValueError: If the policy name is not recognized
    """
    policies = {
        "BUG": bug_policy,
        "FIFO": fifo_policy,
        "LRU": lru_policy
    }

    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Available policies: {', '.join(policies.keys())}")

    return policies[policy_name]


def bug_policy(cache, file_stats):
    """
    Bug-based cache replacement policy.
    Evicts the file with the fewest bug fixes.

    Args:
        cache (list): The current cache of files
        file_stats (dict): Dictionary of file statistics

    Returns:
        str: The file to evict from the cache
    """
    min_bugs = float('inf')
    file_to_remove = None

    for file_path in cache:
        if file_path in file_stats:
            bug_count = file_stats[file_path]["bug_fixes"]
            if bug_count < min_bugs:
                min_bugs = bug_count
                file_to_remove = file_path

    return file_to_remove


def fifo_policy(cache, file_stats):
    """
    First-In-First-Out cache replacement policy.
    Evicts the oldest file in the cache.

    Args:
        cache (list): The current cache of files
        file_stats (dict): Dictionary of file statistics (not used)

    Returns:
        str: The file to evict from the cache
    """
    return cache[0] if cache else None


def lru_policy(cache, file_stats):
    """
    Least Recently Used cache replacement policy.
    Evicts the file that hasn't been modified for the longest time.

    Args:
        cache (list): The current cache of files
        file_stats (dict): Dictionary of file statistics

    Returns:
        str: The file to evict from the cache
    """
    oldest_time = float('inf')
    file_to_remove = None

    for file_path in cache:
        if file_path in file_stats:
            last_modified = file_stats[file_path].get("last_modified", float('inf'))
            if last_modified < oldest_time:
                oldest_time = last_modified
                file_to_remove = file_path

    return file_to_remove