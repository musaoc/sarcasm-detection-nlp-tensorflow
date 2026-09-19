# News Headlines Sarcasm Detection with NLP & TensorFlow

A deep learning NLP model that detects sarcastic nuances in journalistic text by contrasting satirical articles from The Onion against factual news from HuffPost using Bidirectional LSTMs.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/sarcasm-detection-nlp-and-tf-for-beginners)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Natural%20Language%20Processing%20/%20Deep%20Learning-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[News Headlines Sarcasm Detection with NLP & TensorFlow](https://www.kaggle.com/code/lazer999/sarcasm-detection-nlp-and-tf-for-beginners)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Curated clean benchmark data contrasting Onion satire vs Huffington Post reporting to avoid social media noise.
- End-to-end text normalization: Unicode unidecoding, lowercase conversion, and punctuation removal.
- Embedding + Bidirectional LSTM architecture tailored for detecting ironic tone and lexical incongruity.
- Comprehensive training validation curves and custom headline inference utility.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[News Headlines] --> B[Text Normalization & Cleansing]
    B --> C[Tokenization & Padded Sequences]
    C --> D[Embedding Layer]
    D --> E[Bidirectional LSTM]
    E --> F[Dense Output & Sarcasm Classification]
```

---

## Repository Structure

```plaintext
sarcasm-detection-nlp-tensorflow/
├── notebooks/
│   └── sarcasm-detection-nlp-tensorflow.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/sarcasm-detection-nlp-tensorflow.git
cd sarcasm-detection-nlp-tensorflow
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/sarcasm-detection-nlp-tensorflow.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [News Headlines Sarcasm Detection with NLP & TensorFlow](https://www.kaggle.com/code/lazer999/sarcasm-detection-nlp-and-tf-for-beginners)

If you found this project helpful or insightful, please consider starring the repository ⭐!
