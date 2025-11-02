# Speech-to-Text using Hidden Markov Models with Gaussian Emissions (GMM-HMM)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)
[![Kaldi](https://img.shields.io/badge/Kaldi-ASR-orange.svg)](https://kaldi-asr.org/)

An Automatic Speech Recognition (ASR) system implementing Gaussian Mixture Model-Hidden Markov Model (GMM-HMM) architecture for audio transcription, developed as part of the Probabilistic Reasoning course (22AIE301).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training Pipeline](#model-training-pipeline)
- [Results](#results)
- [Course Mapping](#course-mapping)
- [Contributors](#contributors)
- [References](#references)

## 🎯 Overview

This project implements a complete ASR pipeline using GMM-HMM models for Tamil language speech recognition. The system extracts Mel-Frequency Cepstral Coefficients (MFCC) features from audio, models temporal patterns with Hidden Markov Models, and uses Gaussian Mixture Models for acoustic feature representation.

### Key Objectives

- Build an accurate audio-to-text transcription system using GMM-HMM
- Extract and utilize MFCC features for speech representation (39-dimensional)
- Model temporal speech patterns with HMM and acoustic features with GMM
- Implement Baum-Welch algorithm for parameter estimation
- Apply Viterbi decoding for transcription

## ✨ Features

- **Multi-stage Acoustic Modeling**: Progressive training from monophone to triphone models
- **Feature Extraction**: 39-D MFCC features with delta and delta-delta coefficients
- **Advanced Transformations**: LDA+MLLT for feature optimization
- **Language Model Integration**: Trigram language model with Witten-Bell smoothing
- **Kaldi Framework**: Industry-standard ASR toolkit implementation
- **Low-resource Language Support**: Optimized for Tamil language recognition

## 🏗️ System Architecture

### Complete ASR Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPEECH-TO-TEXT GMM-HMM ASR SYSTEM                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ INPUT STAGE                                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Audio (MP3, 48kHz) ──────► Audio Preprocessing ──────► WAV (16kHz, mono)
│                                  • Format conversion                         │
│                                  • Resampling                                │
│                                  • Normalization                             │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────┐
│ FEATURE EXTRACTION STAGE                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Windowing (25ms frames, 10ms shift)                                         │
│         │                                                                    │
│         ├─► FFT ─► Mel Filterbank ─► Log ─► DCT ─► MFCC (13 coefficients)  │
│         │                                                                    │
│         ├─► Compute Δ (Delta) ──────────────────► 13 coefficients          │
│         │                                                                    │
│         └─► Compute ΔΔ (Delta-Delta) ────────────► 13 coefficients          │
│                                                                              │
│  Combined Feature Vector: 39 dimensions (13 + 13Δ + 13ΔΔ)                   │
│         │                                                                    │
│         └─► CMVN Normalization (mean=0, variance=1)                          │
│                                                                              │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────┐
│ ACOUSTIC MODEL TRAINING (Baum-Welch EM Algorithm)                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ Stage 1: MONOPHONE MODEL (Context-Independent)                  │        │
│  │  • States: 112 PDFs                                             │        │
│  │  • Gaussians: 986                                               │        │
│  │  • Context: Single phone                                        │        │
│  │  • WER: 8.0% | Training: ~30 min                                │        │
│  └────────────────────┬────────────────────────────────────────────┘        │
│                       │ Alignment                                           │
│  ┌────────────────────▼────────────────────────────────────────────┐        │
│  │ Stage 2: TRIPHONE MODEL (Context-Dependent)                     │        │
│  │  • States: 456 PDFs                                             │        │
│  │  • Gaussians: 10,039 (~22/state)                                │        │
│  │  • Context: 3 phones (L-C-R)                                    │        │
│  │  • WER: 3.2% | Training: ~2 hours                               │        │
│  │  • Improvement: 60% WER reduction ✓                             │        │
│  └────────────────────┬────────────────────────────────────────────┘        │
│                       │ Alignment                                           │
│  ┌────────────────────▼────────────────────────────────────────────┐        │
│  │ Stage 3: LDA+MLLT MODEL (Feature Transform)                     │        │
│  │  • States: 528 PDFs                                             │        │
│  │  • Gaussians: 13,867 (~26/state)                                │        │
│  │  • Features: 40-D (LDA transformation)                          │        │
│  │  • WER: 3.0% | Training: ~3 hours                               │        │
│  │  • Improvement: 62.5% total WER reduction ✓✓                    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  GMM Parameters: {μₖ, Σₖ, wₖ} learned via EM                                │
│  HMM Parameters: {π, A, B} learned via Baum-Welch                            │
│                                                                              │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────┐
│ LANGUAGE MODEL & LEXICON                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐     ┌─────────────────────┐                        │
│  │  LEXICON (L.fst)    │     │  LANGUAGE MODEL     │                        │
│  │                     │     │  (G.fst)            │                        │
│  │  Words → Phonemes   │     │                     │                        │
│  │  அஃதறம் → /a/ /ḵ/   │     │  Trigram Model      │                        │
│  │  /t/ /a/ /ṟ/ /a/    │     │  P(w₃|w₁,w₂)        │                        │
│  │  /m/                │     │                     │                        │
│  │                     │     │  Witten-Bell        │                        │
│  │  12,000+ entries    │     │  Smoothing          │                        │
│  └─────────────────────┘     └─────────────────────┘                        │
│                                                                              │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────┐
│ DECODING STAGE (Viterbi Algorithm)                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Weighted Finite-State Transducer (WFST) Composition:                       │
│                                                                              │
│         H ∘ C ∘ L ∘ G = HCLG.fst                                             │
│         │   │   │   │                                                        │
│         │   │   │   └─► Grammar (Language Model)                             │
│         │   │   └─────► Lexicon (Pronunciation Dictionary)                   │
│         │   └─────────► Context-dependency (Triphone contexts)               │
│         └─────────────► HMM structure (Acoustic topology)                    │
│                                                                              │
│  Viterbi Beam Search:                                                        │
│    • Finds most likely word sequence W* = argmax P(W|O)                     │
│    • Combines: Acoustic score + Language score + Pronunciation score        │
│    • Beam pruning for efficiency                                            │
│                                                                              │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────┐
│ OUTPUT STAGE                                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Transcribed Text Output                                                     │
│  • Word sequence with confidence scores                                      │
│  • Timestamp alignments (optional)                                           │
│  • WER: 3.0% (97% accuracy) ✓✓✓                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Framework

#### GMM-HMM Joint Probability Model

```
P(O, Q | λ) = P(Q | λ) · P(O | Q, λ)
            = π(q₁) · ∏ᵗ₌₂ᵀ a(qₜ₋₁, qₜ) · ∏ᵗ₌₁ᵀ bqₜ(oₜ)

Where:
  O = {o₁, o₂, ..., oₜ}  : Observation sequence (MFCC features)
  Q = {q₁, q₂, ..., qₜ}  : Hidden state sequence (phonemes)
  λ = {π, A, B}          : HMM parameters
  
  π(qᵢ)    : Initial state probability
  a(i,j)   : Transition probability P(qₜ=j | qₜ₋₁=i)
  bⱼ(oₜ)   : Observation probability P(oₜ | qₜ=j) ← GMM
```

#### GMM Observation Probability

```
P(oₜ | qₜ=j) = ∑ₖ₌₁ᴷ wⱼₖ · 𝒩(oₜ | μⱼₖ, Σⱼₖ)

Where:
  K      : Number of Gaussian components
  wⱼₖ    : Mixture weight for component k in state j
  μⱼₖ    : Mean vector
  Σⱼₖ    : Covariance matrix
  𝒩()    : Multivariate Gaussian distribution
```

### Component Breakdown

| Component | Function | Input | Output |
|-----------|----------|-------|--------|
| **GMM** | Models observation likelihoods | MFCC features | P(oₜ \| qₜ) |
| **HMM** | Models temporal structure | State sequence | P(qₜ \| qₜ₋₁) |
| **MFCC** | Feature extraction | Audio waveform | 39-D vectors |
| **LDA+MLLT** | Feature transformation | 39-D features | 40-D features |
| **Lexicon** | Word-to-phoneme mapping | Words | Phoneme sequences |
| **Language Model** | Context prediction | Word history | P(wₜ \| wₜ₋₂, wₜ₋₁) |
| **Viterbi Decoder** | Best path search | Features + Models | Text output |

### Data Flow Summary

```
Audio (48kHz) → Preprocessing → WAV (16kHz) → MFCC Extraction → 39-D Features
→ CMVN Normalization → GMM-HMM Training (3 stages) → Acoustic Models
→ WFST Composition (H∘C∘L∘G) → Viterbi Decoding → Transcribed Text (97% accuracy)
```

## 📊 Dataset

**Common Voice Corpus 6.1 - Tamil**

| Parameter | Value |
|-----------|-------|
| Version | cv-corpus-6.1-2020-12-11 |
| Release Date | December 11, 2020 |
| Language | Tamil (ta) |
| Total Duration | 24 hours |
| Validated Duration | 14 hours (58.3%) |
| Format | MP3 (48kHz) → WAV (16kHz mono) |
| Bit Depth | 16-bit |
| Training Clips | 12,652 utterances |
| Test Clips | 1,781 utterances |
| Total Validated | 14,433+ clips |
| Speakers | Crowdsourced (diverse) |
| Download Size | ~1-2 GB (tar.gz) |

**Download**: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

## 🔧 Installation

### Prerequisites

- **Operating System**: Ubuntu 20.04 (or WSL2 on Windows)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Compiler**: GCC 7.5+

### Step 1: Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Build tools
sudo apt-get install -y gcc g++ make autoconf automake libtool \
    cmake git wget unzip sox

# Math libraries
sudo apt-get install -y libatlas-base-dev liblapack-dev libblas-dev

# Audio processing
sudo apt-get install -y sox ffmpeg libsndfile1-dev

# Python dependencies
sudo apt-get install -y python3 python3-pip
pip3 install numpy scipy matplotlib
```

### Step 2: Install Kaldi

```bash
# Clone Kaldi repository
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi

# Build tools
cd tools
make -j $(nproc)

# Install SRILM (for language modeling)
# Download from: http://www.speech.sri.com/projects/srilm/download.html
# Extract and compile according to SRILM documentation

# Build Kaldi source
cd ../src
./configure --shared
make depend -j $(nproc)
make -j $(nproc)
```

### Step 3: Setup Project

```bash
# Clone this repository
git clone https://github.com/yourusername/gmm-hmm-asr.git
cd gmm-hmm-asr

# Create data directories
mkdir -p data/train data/test data/local/dict data/local/lang

# Convert audio files to required format
for file in raw_audio/*.mp3; do
    sox "$file" -r 16000 -c 1 -b 16 "data/wav/$(basename $file .mp3).wav"
done
```

## 🚀 Usage

### Data Preparation

```bash
# Prepare data files
python scripts/prepare_data.py \
    --corpus_dir /path/to/common_voice \
    --output_dir data/

# Prepare lexicon
python scripts/prepare_dict.py \
    --train_text data/train/text \
    --dict_dir data/local/dict
```

### Feature Extraction

```bash
# Extract MFCC features
steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
    data/train exp/make_mfcc/train mfcc

steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
    data/test exp/make_mfcc/test mfcc

# Compute CMVN statistics
steps/compute_cmvn_stats.sh \
    data/train exp/make_mfcc/train mfcc
```

### Model Training

```bash
# 1. Train Monophone Model
steps/train_mono.sh --nj 4 --cmd "run.pl" \
    data/train data/lang exp/mono

# 2. Align with Monophone
steps/align_si.sh --nj 4 --cmd "run.pl" \
    data/train data/lang exp/mono exp/mono_ali

# 3. Train Triphone Model
steps/train_deltas.sh --cmd "run.pl" 2000 10000 \
    data/train data/lang exp/mono_ali exp/tri1

# 4. Align with Triphone
steps/align_si.sh --nj 4 --cmd "run.pl" \
    data/train data/lang exp/tri1 exp/tri1_ali

# 5. Train LDA+MLLT Model
steps/train_lda_mllt.sh --cmd "run.pl" 2500 15000 \
    data/train data/lang exp/tri1_ali exp/tri2b
```

### Language Model Creation

```bash
# Build trigram language model
python scripts/prepare_lm.py \
    --text data/train/text \
    --output data/local/lm/trigram.arpa

# Convert to FST format
utils/format_lm.sh data/lang data/local/lm/trigram.arpa \
    data/local/dict/lexicon.txt data/lang_test
```

### Decoding

```bash
# Create decoding graph
utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph

# Decode test set
steps/decode.sh --nj 4 --cmd "run.pl" \
    exp/tri2b/graph data/test exp/tri2b/decode_test

# Calculate Word Error Rate (WER)
grep WER exp/tri2b/decode_test/wer_* | utils/best_wer.sh
```

### Transcribe New Audio

```bash
# Transcribe a single audio file
python scripts/transcribe.py \
    --audio input.wav \
    --model exp/tri2b \
    --graph exp/tri2b/graph \
    --output transcription.txt
```

## 🔬 Model Training Pipeline

### Acoustic Models

#### 1. Monophone Model (mono)
- **Type**: Context-independent (flat-start)
- **States**: 112 PDFs
- **Gaussians**: 986
- **Context Width**: 1
- **Training Iterations**: 40
- **Purpose**: Baseline model for alignment

#### 2. Triphone Model (tri1)
- **Type**: Context-dependent triphones
- **States**: 456 PDFs
- **Gaussians**: 10,039 (~22 per state)
- **Context Width**: 3 (left + center + right phone)
- **Features**: 39-D (MFCC + Δ + ΔΔ)
- **Purpose**: Context modeling

#### 3. LDA+MLLT Model (tri2b)
- **Type**: Feature-transformed triphone
- **States**: 528 PDFs
- **Gaussians**: 13,867 (~26 per state)
- **Context Width**: 3
- **Feature Dimension**: 40-D (after LDA transform)
- **Purpose**: Feature transformation and dimensionality reduction

### Training Algorithm: Baum-Welch (EM)

```
1. Initialization: Set initial HMM parameters (π, A, B)
2. E-Step (Expectation):
   - Use Forward-Backward algorithm
   - Compute state occupation probabilities γ_t(i)
   - Compute state transition probabilities ξ_t(i,j)
3. M-Step (Maximization):
   - Update transition probabilities: A = f(ξ)
   - Update observation probabilities: B = f(γ)
   - Update GMM parameters: μ, Σ, w = f(γ)
4. Iterate until convergence (log-likelihood change < threshold)
```

### Decoding Algorithm: Viterbi

```
Input: Observation sequence O = {o_1, o_2, ..., o_T}
Output: Best state sequence Q* = {q_1*, q_2*, ..., q_T*}

1. Initialization:
   δ_1(i) = π_i · b_i(o_1)
   ψ_1(i) = 0

2. Recursion (t = 2 to T):
   δ_t(j) = max_i [δ_{t-1}(i) · a_{ij}] · b_j(o_t)
   ψ_t(j) = argmax_i [δ_{t-1}(i) · a_{ij}]

3. Termination:
   P* = max_i [δ_T(i)]
   q_T* = argmax_i [δ_T(i)]

4. Backtracking (t = T-1 to 1):
   q_t* = ψ_{t+1}(q_{t+1}*)
```

## 📈 Results

### Model Performance

| Model | States | Gaussians | WER (%) | Training Time | Relative Improvement |
|-------|--------|-----------|---------|---------------|---------------------|
| Monophone (mono) | 112 | 986 | 8.0% | ~30 min | Baseline |
| Triphone (tri1) | 456 | 10,039 | 3.2% | ~2 hours | 60% reduction |
| LDA+MLLT (tri2b) | 528 | 13,867 | **3.0%** | ~3 hours | **62.5% reduction** |

### Performance Metrics

- **Best WER Achieved**: 3.0% (LDA+MLLT model)
- **Accuracy**: 97.0% word-level recognition
- **Feature Dimensions**: 39-D MFCC → 40-D (after LDA transformation)
- **Context Width**: 3 phones (triphone modeling)
- **Beam Width**: Optimized for accuracy-speed tradeoff
- **Total Training Time**: ~5.5 hours for complete pipeline

### Key Observations

1. **Significant Progressive Improvement**: 
   - Monophone → Triphone: 60% WER reduction (8.0% → 3.2%)
   - Triphone → LDA+MLLT: Additional 6.25% improvement (3.2% → 3.0%)
   
2. **Context Modeling Impact**: Triphone models dramatically outperform context-independent monophones

3. **Feature Transformation Benefits**: LDA+MLLT provides optimal feature representation and dimensionality reduction

4. **Low-resource Language Success**: Achieving 97% accuracy on Tamil demonstrates effective adaptation for low-resource languages

5. **Production-Ready Performance**: 3.0% WER is competitive with modern ASR systems for similar datasets

## 🎓 Course Mapping

### CO2: Apply Bayesian & Markov Networks
- **HMM**: Transition probability modeling (Markov Network)
- **GMM**: Observation probability modeling (Bayesian mixture)

### CO3: Apply Probabilistic Reasoning for Decision Making
- **Observation Probability**: GMM gradient as input for HMM
- **Transition Probability**: Modeled with HMM for sequential reasoning
- **Lexicon Integration**: Enables complex decision-making in ASR

### CO4: Apply Computational Tools for Probabilistic Models
- **EM Algorithm**: Training GMM parameters
- **Baum-Welch Algorithm**: Training HMM parameters
- **ASR Pipeline**: Combines GMM+HMM+Lexicon with modern frameworks

## 👥 Contributors

**Group 5 - Probabilistic Reasoning (22AIE301)**

- **Antonio Roger** (CB.SC.U4AIE23104)
- **Adarsh Pradeep** (CB.SC.U4AIE23109)
- **Mohan Raj** (CB.SC.U4AIE23147)
- **Naresh Kumar V** (CB.SC.U4AIE23165)

## 📚 References

1. Farheen Fauziya & Geeta Nijhawan. "A Comparative Study of Phoneme Recognition using GMM-HMM and ANN based Acoustic Modeling." *International Journal of Computer Applications*, Volume 98, No. 6, July 2014.

2. Abdelmadjid Benmachiche & Amina Makhlouf. "Optimization of Hidden Markov Model With Gaussian Mixture Densities." *WSEAS Transactions on Signal Processing*, Volume 15, 2019.

3. Yi Liu, Liang He, Weiqiang Zhang, Jia Liu, Michael T. Johnson. "Investigation of Frame Alignments for GMM-based Digit-prompted Speaker Verification." arXiv:1710.10436, 2017.

4. Edmondo Trentin & Marco Gori. "Robust Combination of Neural Networks and Hidden Markov Models for Speech Recognition." *IEEE Transactions on Neural Networks*, Volume 14, Issue 6, November 2003.

5. Liang Lu. "Subspace Gaussian Mixture Models for Automatic Speech Recognition." PhD Thesis, University of Edinburgh, 2013.

6. Kaldi ASR Documentation: https://kaldi-asr.org/doc/

7. Common Voice Dataset: https://commonvoice.mozilla.org/


## 🙏 Acknowledgments

- **Kaldi Speech Recognition Toolkit** for providing the ASR framework
- **Mozilla Common Voice** for the Tamil language dataset
- **Course Instructor** for guidance on probabilistic reasoning concepts
- **SRI International** for the SRILM language modeling toolkit

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the contributors through the university portal.

---

**Note**: This project was developed as part of the academic curriculum for educational purposes. The implementation follows standard ASR practices and leverages established frameworks for reproducibility.
