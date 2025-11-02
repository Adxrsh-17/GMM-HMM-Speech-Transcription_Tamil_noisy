#!/bin/bash

# generate_readme.sh - Auto-generates professional README.md with LaTeX math
# Usage: bash generate_readme.sh [--repo-url URL] [--output README.md]

REPO_URL="https://github.com/your-username/tamil-gmm-hmm-asr"
OUTPUT_FILE="README.md"

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
      exit 1
      ;;
  esac
done

cat > "$OUTPUT_FILE" << 'EOF'
# 🎤 Speech-to-Text using Hidden Markov Models with Gaussian Emissions (GMM-HMM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Kaldi ASR](https://img.shields.io/badge/Kaldi-ASR-green.svg)](http://kaldi-asr.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-complete-brightgreen)](https://github.com)

---

## 📊 Project Workflow

