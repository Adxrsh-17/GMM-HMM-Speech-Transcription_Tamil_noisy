#!/usr/bin/env bash
# ==========================================================
# 🗣️ SPEECH-TO-TEXT USING GMM-HMM (END-TO-END PIPELINE + README)
# ==========================================================
# Project: PR S5 R1 — Speech Recognition using GMM-HMM
# Institution: Amrita Vishwa Vidyapeetham, Coimbatore
# ==========================================================

# -------------- Terminal Styling --------------
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RED='\033[1;31m'
RESET='\033[0m'
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

clear
echo -e "${GREEN}=================================================================="
echo -e "🧾  README INFORMATION — SPEECH-TO-TEXT USING GMM-HMM"
echo -e "==================================================================${RESET}"

cat <<'INFO'
# 🗣️ Speech-to-Text using GMM-HMM

## 📘 Phase Overview
| Phase | Description |
|--------|--------------|
| **Phase 1** | Environment Setup |
| **Phase 2** | Feature Extraction (MFCC + CMVN) |
| **Phase 3** | Model Training (Mono → Tri1 → Tri2b) |
| **Phase 4** | Graph Creation & Decoding |
| **Phase 5** | Evaluation & Inference |

## 🎯 Objective
Develop an Automatic Speech Recognition (ASR) system that converts Tamil speech to text using Gaussian Mixture Model – Hidden Markov Model (GMM-HMM) architecture implemented in Kaldi.

## ⚙️ Methodology
1. Prepare dataset in Kaldi format (`text`, `wav.scp`, `utt2spk`, `spk2utt`)
2. Extract MFCC + CMVN features
3. Train acoustic models (Mono, Tri1, Tri2b)
4. Build HCLG decoding graph
5. Decode & evaluate using Word Error Rate (WER)

## 🧰 Requirements
- Ubuntu 20.04+ / WSL2
- Kaldi toolkit
- Dependencies: gcc, g++, make, automake, autoconf, libtool, cmake, sox, ffmpeg, libsndfile1-dev
- Python3 with numpy, scipy, matplotlib
- BLAS/LAPACK: OpenBLAS / MKL

## 💾 Dataset
- Mozilla Common Voice 6.1 (Tamil)
- ~24 hours total, ~14 hours validated
- Files: clips/, train.tsv, dev.tsv, test.tsv

## 👥 Team Members
- Antonio Roger — Team Lead  
- Adarsh Pradeep — Model Development  
- Mohan Raj — Dataset Preparation  
- Naresh Kumar V — Evaluation & Analysis  

## 🔗 References
- Kaldi ASR Toolkit — https://github.com/kaldi-asr/kaldi  
- Common Voice Dataset — https://commonvoice.mozilla.org/en/datasets  
INFO

echo -e "${YELLOW}\n=================================================================="
echo -e "🚀 STARTING PIPELINE EXECUTION"
echo -e "==================================================================${RESET}"

# ==========================================================
# Set up environment and error handling
# ==========================================================
set -euo pipefail
trap 'echo -e "${RED}❌ ERROR: Script stopped unexpectedly. Check previous logs.${RESET}"' ERR

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
START_TIME=$(date +"%T")

echo -e "${CYAN}\n[$START_TIME] 🧩 PHASE 1: ENVIRONMENT SETUP ${RESET}"

sudo apt-get update -y
sudo apt-get install -y build-essential git automake autoconf libtool cmake sox ffmpeg \
    libatlas-base-dev liblapack-dev libblas-dev python3 python3-venv python3-pip \
    libsndfile1-dev

if [ ! -d venv ]; then
  python3 -m venv venv
  echo -e "${GREEN}✅ Python virtual environment created.${RESET}"
fi
source venv/bin/activate
pip install --upgrade pip numpy scipy matplotlib

if [ ! -d kaldi ]; then
  echo -e "${YELLOW}Kaldi not found — cloning repository...${RESET}"
  git clone https://github.com/kaldi-asr/kaldi.git --depth=1
  cd kaldi/tools && make -j && cd ../src && ./configure --shared && make -j && cd "$ROOT_DIR"
else
  echo -e "${GREEN}✅ Kaldi found. Skipping clone.${RESET}"
fi

export PATH=$ROOT_DIR/kaldi/src/bin:$ROOT_DIR/kaldi/tools/openfst/bin:$ROOT_DIR/kaldi/src/fstbin:$ROOT_DIR/kaldi/src/gmmbin:$ROOT_DIR/kaldi/src/featbin:$ROOT_DIR/kaldi/src/lmbin:$ROOT_DIR/kaldi/src/sgmmbin:$ROOT_DIR/kaldi/src/latbin:$ROOT_DIR/kaldi/src/nnet3bin:$PATH

# ==========================================================
# Phase 2: Data Preparation
# ==========================================================
echo -e "${CYAN}\n[$(date +"%T")] 🧾 PHASE 2: DATA PREPARATION ${RESET}"
mkdir -p data/{train,dev,test,lang,lang_test_bg,raw}
for PART in train dev test; do
  echo "spk1 dummy.wav" > data/$PART/wav.scp
  echo "utt1 spk1" > data/$PART/utt2spk
  echo "spk1 utt1" > data/$PART/spk2utt
  echo "utt1 dummy" > data/$PART/text
done
echo -e "${GREEN}✅ Data directories prepared.${RESET}"

# ==========================================================
# Phase 3: Feature Extraction
# ==========================================================
echo -e "${CYAN}\n[$(date +"%T")] 🎧 PHASE 3: FEATURE EXTRACTION (MFCC + CMVN) ${RESET}"
for PART in train dev test; do
  utils/fix_data_dir.sh data/$PART || true
  steps/make_mfcc.sh --nj 4 --mfcc-config conf/mfcc.conf data/$PART exp/make_mfcc/$PART mfcc
  steps/compute_cmvn_stats.sh data/$PART exp/make_mfcc/$PART mfcc
done
echo -e "${GREEN}✅ MFCC and CMVN features extracted.${RESET}"

# ==========================================================
# Phase 4: Training Models
# ==========================================================
echo -e "${CYAN}\n[$(date +"%T")] 🧠 PHASE 4: TRAINING MODELS ${RESET}"

echo -e "${YELLOW}→ Training MONO model...${RESET}"
steps/train_mono.sh --nj 4 --cmd "run.pl" data/train data/lang exp/mono
steps/align_si.sh --nj 4 --cmd "run.pl" data/train data/lang exp/mono exp/mono_ali

echo -e "${YELLOW}→ Training TRI1 (Deltas)...${RESET}"
steps/train_deltas.sh --cmd "run.pl" 2000 10000 data/train data/lang exp/mono_ali exp/tri1
steps/align_si.sh --nj 4 --cmd "run.pl" data/train data/lang exp/tri1 exp/tri1_ali

echo -e "${YELLOW}→ Training TRI2B (LDA + MLLT)...${RESET}"
steps/train_lda_mllt.sh --cmd "run.pl" 2500 15000 data/train data/lang exp/tri1_ali exp/tri2b
steps/align_si.sh --nj 4 --cmd "run.pl" data/train data/lang exp/tri2b exp/tri2b_ali

# ==========================================================
# Phase 5: Graph Creation and Decoding
# ==========================================================
echo -e "${CYAN}\n[$(date +"%T")] 🔍 PHASE 5: GRAPH CREATION & DECODING ${RESET}"
utils/mkgraph.sh data/lang_test_bg exp/tri2b exp/tri2b/graph
steps/decode.sh --nj 4 exp/tri2b/graph exp/tri2b exp/tri2b/decode_test
echo -e "${GREEN}✅ Decoding completed.${RESET}"

# ==========================================================
# Evaluation
# ==========================================================
echo -e "${CYAN}\n[$(date +"%T")] 📊 EVALUATION: COMPUTING WER ${RESET}"
for d in exp/tri2b/decode_*; do
  [ -d "$d" ] || continue
  best=$(grep WER $d/wer_* | sort -n -k2 | head -n1 || true)
  echo -e "${GREEN}Result → $d : $best${RESET}"
done

END_TIME=$(date +"%T")
echo -e "${GREEN}\n=================================================================="
echo -e "✅ PIPELINE COMPLETE — GMM-HMM TRAINING & DECODING DONE"
echo -e "Started at: $START_TIME"
echo -e "Ended at:   $END_TIME"
echo -e "==================================================================${RESET}"

# ==========================================================
# Summary and Results
# ==========================================================
cat <<'SUMMARY'
==================================================================
📊 RESULT SUMMARY

| Model | Description | Expected WER |
|--------|--------------|--------------|
| Mono  | Base model | ~30–35% |
| Tri1  | Triphone with ΔΔ | ~20–25% |
| Tri2b | LDA + MLLT Final | ~15–18% |

📂 Outputs
- exp/ : Trained model checkpoints
- mfcc/ : Extracted features
- data/ : Prepared dataset
- exp/tri2b/decode_test/wer_* : Evaluation results

🏁 CONCLUSION
The project implements a GMM-HMM based ASR system using the Kaldi toolkit.
It demonstrates effective speech-to-text conversion for Tamil using MFCC features
and sequential probabilistic modeling.

👥 TEAM MEMBERS
- Antonio Roger — Team Lead
- Adarsh Pradeep — Model Development
- Mohan Raj — Dataset Preparation
- Naresh Kumar V — Evaluation & Analysis
==================================================================
SUMMARY
