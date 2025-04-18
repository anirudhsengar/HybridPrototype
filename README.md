
# Hybrid Bug Predictor

A hybrid approach to software defect prediction combining temporal patterns (FixCache/BugCache) with code metrics analysis (REPD) for improved bug prediction accuracy.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-EPL--2.0-blue.svg)](https://www.eclipse.org/legal/epl-2.0/)

## Overview

This project implements a novel hybrid approach to software defect prediction by combining two complementary techniques:

1. **FixCache Algorithm**: Analyzes version control history to identify temporal patterns in bug fixes
2. **Reconstruction Error Probability Distribution (REPD)**: Uses code metrics and autoencoders to detect anomalous code structures

The hybrid approach dynamically weights both models based on repository characteristics, providing more accurate predictions than either method alone.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for repository analysis)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/anirudhsengar/hybrid-bug-predictor.git
cd hybrid-bug-predictor

# Install the package
pip install .
```

### Installation with Optional Dependencies

```bash
# Install with visualization capabilities
pip install .[visualization]

# Install with development tools
pip install .[dev]

# Install all dependencies
pip install .[full]
```

## Usage

### Basic Usage

```python
from HybridPrototype.hybrid.predictor import HybridPredictor

# Initialize the predictor with a repository path
predictor = HybridPredictor(repo_path="/path/to/git/repository")

# Analyze the repository
predictor.analyze_repository()

# Get top risky files
top_files = predictor.get_top_risky_files(limit=10)

# Print results
for i, (file_path, score) in enumerate(top_files, 1):
    print(f"{i}. {file_path}: {score:.4f}")
```

### Command Line Interface

```bash
# Analyze a repository
bugpredictor analyze /path/to/git/repository --cache-size 0.1 --top 20

# Visualize results
bugpredictor visualize /path/to/git/repository --output-dir ./results
```

## How It Works

### The Hybrid Approach

1. **Repository Analysis**: The unified repository analyzer extracts commit history, identifies bug fixes, and collects file metrics.

2. **FixCache Prediction**: Implements the BugCache/FixCache algorithm to identify files likely to contain defects based on temporal and spatial locality.

3. **REPD Analysis**: Uses code metrics and autoencoder reconstruction error to identify anomalous code structures likely to contain defects.

4. **Dynamic Weighting**: Calculates optimal weights for each prediction approach based on repository characteristics.

5. **Combined Prediction**: Merges predictions from both approaches using the calculated weights to produce a final risk score for each file.

## Project Structure

```
HybridPrototype/
├── __init__.py
├── repository.py    
├── fixcache/        
├── repd/           
├── hybrid/         
│   ├── __init__.py
│   ├── predictor.py  <-- This contains the HybridPredictor class
│   └── weighting.py
└── visualization.py
└── setup.py
```

## Dependencies

- **Core Dependencies**:
  - `gitpython`: Git repository interaction
  - `numpy` & `pandas`: Data processing
  - `scikit-learn`: Machine learning components

- **Optional Dependencies**:
  - `matplotlib` & `seaborn`: Visualization
  - `pytest`: Testing framework

## Research Background

This implementation is based on research in software defect prediction:

- FixCache/BugCache approach from "Using History to Improve Mobile Application Security" (Kim et al.)
- REPD approach from "Bug Prediction Through Reconstruction Error in Neural Networks" (Various authors)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Eclipse Public License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of the Google Summer of Code 2025 program with the Eclipse Foundation. It adapts and combines approaches from the following prototypes:

- [FixCachePrototype](https://github.com/anirudhsengar/FixCachePrototype)
- [REPDPrototype](https://github.com/anirudhsengar/REPDPrototype)

## Author

- **Anirudh Sengar** - [GitHub Profile](https://github.com/anirudhsengar)