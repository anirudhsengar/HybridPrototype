#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="hybrid-bug-predictor",
    version="0.1.0",
    description="Hybrid approach to software defect prediction",
    author="anirudhsengar",
    author_email="anirudhsengar3@gmail.com",
    packages=find_packages(),
    install_requires=[
        "gitpython>=3.1.0",
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)