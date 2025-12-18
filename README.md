
# Codemixed Text-to-Speech using Bilingual Phoneme Embeddings

## Overview

This project explores **text-to-speech (TTS) synthesis for codemixed text**, where multiple languages appear within the same utterance (e.g., Hindi–English). Standard monolingual phoneme representations struggle in this setting due to abrupt phonetic and linguistic transitions.

The core contribution of this work is the design and evaluation of **custom bilingual phoneme embeddings** integrated into a neural TTS pipeline, aimed at improving intelligibility and pronunciation consistency for codemixed speech.

This work was conducted as part of a **research internship in a FLAME, IIITD**.

---

## Problem Statement

Most TTS systems assume:
- a single language per utterance
- language-specific phoneme inventories
- stable phonotactic rules

In codemixed speech, these assumptions break down:
- phonemes from different languages co-occur
- cross-language transitions introduce ambiguity
- monolingual embeddings fail to generalize

The objective of this project is to **improve speech quality for codemixed inputs** by learning representations that explicitly model bilingual phonetic structure.

---

## Approach

### 1. Baseline Evaluation
- Trained and evaluated standard TTS architectures using **monolingual phoneme embeddings** on codemixed data.
- Identified consistent pronunciation degradation at language-switch boundaries.

### 2. Bilingual Phoneme Embedding Design
- Designed **custom bilingual phoneme embeddings** that jointly encode phonemes from both languages.
- Explicitly modeled shared and language-specific phonetic features.
- Integrated these embeddings into the acoustic modeling stage.

### 3. Model Integration
- Embedded the bilingual phoneme representations into a **Neural Codec Language Model (NCLM)** based TTS pipeline.
- Maintained identical training setups across baselines for controlled comparison.

### 4. Evaluation
- Compared baseline vs. bilingual embeddings using:
  - intelligibility
  - pronunciation consistency
  - subjective audio quality
- Conducted qualitative error analysis focused on codemixed transition points.

---

## Dataset

- Codemixed speech dataset containing utterances with multiple languages in a single sentence.
- Data split into train / validation / test sets.
- Language labels and phoneme sequences derived during preprocessing.

---

## Key Results

- Bilingual phoneme embeddings improved pronunciation stability at language-switch boundaries.
- Reduced mispronunciation of borrowed words and named entities.
- Gains were most visible in utterances with frequent language alternation.

---

## Technical Highlights

- Speech representation learning
- Custom embedding design
- Neural TTS pipelines
- Controlled experimental evaluation
- Linguistic analysis of model failure cases

---

## Limitations

- Evaluation primarily qualitative due to limited standardized metrics for codemixed TTS.
- Dataset size constrained by availability of annotated codemixed speech.
- Results may not generalize to all language pairs without retraining embeddings.

---



