#!/bin/bash

# generate_readme.sh
# Auto-generates professional README.md for GitHub repository
# Usage: bash generate_readme.sh [--repo-url URL] [--output README.md]

# Default values
REPO_URL="https://github.com/your-username/tamil-gmm-hmm-asr"
OUTPUT_FILE="README.md"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash generate_readme.sh [--repo-url URL] [--output README.md]"
      exit 1
      ;;
  esac
done

# Create README.md with all content
cat > "$OUTPUT_FILE" << 'EOF'
# Speech-to-Text using Hidden Markov Models with Gaussian Emissions (GMM-HMM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Kaldi ASR](https://img.shields.io/badge/Kaldi-ASR-green.svg)](http://kaldi-asr.org/)

## Overview

A comprehensive Automatic Speech Recognition (ASR) system for **low-resource Tamil** audio transcription using Hidden Markov Models (HMM) with Gaussian Mixture Model (GMM) acoustic emissions. This project implements a complete end-to-end pipeline combining probabilistic modeling, signal processing, and Bayesian inference for accurate speech-to-text conversion.

**Key Achievement:** Successfully trained context-dependent acoustic models (monophone, triphone, LDA+MLLT) achieving optimal performance with 528 PDFs and 13,867 Gaussians.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technical Details](#technical-details)
- [Course Mapping](#course-mapping)
- [References](#references)
- [Contributors](#contributors)

---

## Features

✓ **Complete ASR Pipeline** — Feature extraction → Acoustic modeling → Decoding
✓ **GMM-HMM Framework** — Probabilistic modeling of speech acoustics
✓ **Multiple Acoustic Models** — Monophone, triphone, and LDA+MLLT transformations
✓ **Lexicon Integration** — Tamil word-to-phoneme mapping for linguistic constraints
✓ **N-gram Language Model** — Trigram model with Witten-Bell smoothing
✓ **Viterbi Decoding** — Efficient optimal sequence recovery
✓ **Low-Resource Language Support** — Demonstrated on Tamil (cv-corpus-6.1)
✓ **Production-Ready Kaldi Toolkit** — Industry-standard framework with C++ optimization

---

## Architecture

### System Components

