# Speech-To-Text Using Hidden Markov Models with Gaussian Emissions (GMM-HMM) for Audio Transcription

---

## Project Members

- Antonio Roger (CB.SC.U4AIE23104)  
- Adarsh Pradeep (CB.SC.U4AIE23109)  
- Mohan Raj (CB.SC.U4AIE23147)  
- Naresh Kumar V (CB.SC.U4AIE23165)

---

## Objectives

- Build an Automatic Speech Recognition (ASR) system using GMM-HMM for accurate audio-to-text transcription.  
- Extract and utilize Mel-Frequency Cepstral Coefficients (MFCCs) to represent speech features.  
- Model temporal speech patterns with Hidden Markov Models (HMM) combined with Gaussian Mixture Models (GMM).  
- Implement Baum-Welch algorithms for parameter estimation and Viterbi decoding for transcription.

---

## Course Outcomes

| Outcome | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| CO2     | Apply Bayesian & Markov Networks: transition probability modeling with HMM and observation probability modeling with GMM. |
| CO3     | Apply probabilistic reasoning for sequential decision-making and lexicon integration in ASR. |
| CO4     | Employ computational algorithms like EM and Baum-Welch in probabilistic model training.      |

---

## Dataset

We utilize the Mozilla Common Voice dataset, one of the largest and most diverse open-source speech corpora.

- **Source**: [Mozilla Common Voice Dataset](https://commonvoice.mozilla.org/datasets)  
- **Description**: Crowdsourced multilingual speech corpus for training voice recognition models, including Tamil [translate:தமிழ்].  
- **Content**: 24 hours of validated speech data, diverse speakers, 16kHz mono WAV files with transcriptions.  
- **Structure**: Audio clips organized with metadata and validation labels for quality control.  

---

## Key Concepts

- **Acoustic Model (AM)**: Maps acoustic features (MFCCs) to phonemes.  
- **Language Model (LM)**: Statistical models (n-grams) predicting word sequences to improve recognition accuracy.  
- **Lexicon**: Dictionary mapping words to phonemes, bridging AM and LM.  

---

## Methodology Overview

flowchart TD
A[Speech Audio] --> B[MFCC Feature Extraction]
B --> C[GMM-HMM Acoustic & Temporal Modeling]
C --> D[Baum-Welch EM Training]
D --> E[Viterbi Decoding]
E --> F[Transcribed Text]


