# GlitchWitcher: Hybrid Bug Prediction

![Status: Under Development](https://img.shields.io/badge/Status-Under%20Development-yellow)
![License](https://img.shields.io/badge/License-Eclipse%20Public%202.0-blue)

A hybrid approach to software defect prediction combining temporal patterns (BugCache/FixCache) with code metrics analysis (REPD) for improved bug prediction accuracy.

## 🚧 Project Status

This project is currently **under active development** as part of a Google Summer of Code (GSoC) 2025 project with the Eclipse Foundation. The implementation is in its early stages and not yet ready for production use.

## 📖 Overview

GlitchWitcher combines two complementary approaches to bug prediction:

1. **BugCache/FixCache Algorithm**: Analyzes version control history to identify temporal patterns in bug fixes
2. **Reconstruction Error Probability Distribution (REPD)**: Uses code metrics and autoencoders to detect anomalous code structures

The hybrid approach dynamically weights both models based on repository characteristics, providing more accurate predictions than either method alone.

## 🔍 Planned Features

- Unified repository analysis for Git repositories
- Multiple cache policies for temporal prediction
- Code metrics extraction with language-specific support
- Dynamic weight adjustment based on repository characteristics
- GitHub integration for pull request analysis
- Visual reporting of prediction results

## 🛠️ Development Roadmap

- [x] Initial repository setup
- [ ] Core interfaces and data structures
- [ ] Port FixCache implementation from prototype
- [ ] Port REPD implementation from prototype
- [ ] Develop hybrid weighting mechanism
- [ ] Evaluation framework
- [ ] GitHub Actions integration

## 💻 Installation

*Coming soon! The project is not yet ready for installation.*

## 📊 Usage

*Documentation will be provided as the project develops.*

## 🔗 Related Projects

This project builds on two separate prototypes:

- [FixCachePrototype](https://github.com/anirudhsengar/FixCachePrototype): Implementation of the BugCache/FixCache approach
- [REPDPrototype](https://github.com/anirudhsengar/REPDPrototype): Implementation of the REPD model

## 📄 License

Eclipse Public License 2.0

## 👤 Author

Anirudh Sengar (Google Summer of Code 2025)

## 🙏 Acknowledgments

This project is developed as part of the Google Summer of Code 2025 program with the Eclipse Foundation.