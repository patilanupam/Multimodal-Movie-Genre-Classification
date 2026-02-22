# 🎬 Multimodal Movie Genre Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![NLTK](https://img.shields.io/badge/NLP-NLTK%20%7C%20GloVe-green)
![EfficientNet](https://img.shields.io/badge/CNN-EfficientNet--B3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Course:** EAS 510LEC (AI Basics) – Spring 2025

**Authors:** Anupam Patil *(Data Specialist & Image Modelling Lead)* · Nandini Soni *(Text Model & Fusion Lead)*

🌐 **[Live Demo — Genre Genie](https://genregenie.netlify.app/)**

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline at a Glance](#2-pipeline-at-a-glance)
3. [Repository Structure](#3-repository-structure)
4. [Stage 1 — Data Collection](#4-stage-1--data-collection)
5. [Stage 2 — Exploratory Data Analysis](#5-stage-2--exploratory-data-analysis)
6. [Stage 3 — Text-Only Model (NLP)](#6-stage-3--text-only-model-nlp)
   - [NLP Preprocessing Pipeline](#61-nlp-preprocessing-pipeline)
   - [Word Embeddings — GloVe](#62-word-embeddings--glove)
   - [Baseline Text Models Compared](#63-baseline-text-models-compared)
   - [Final Architecture — LSTM](#64-final-architecture--lstm)
   - [Text-Only Evaluation](#65-text-only-evaluation)
7. [Stage 4 — Image-Only Model (CNN)](#7-stage-4--image-only-model-cnn)
   - [Poster Preprocessing](#71-poster-preprocessing)
   - [Baseline Image Models Compared](#72-baseline-image-models-compared)
   - [Final Architecture — EfficientNet-B3](#73-final-architecture--efficientnet-b3)
   - [Classification Modes](#74-classification-modes)
   - [Image-Only Evaluation](#75-image-only-evaluation)
8. [Stage 5 — Multimodal Fusion](#8-stage-5--multimodal-fusion)
9. [Results Summary](#9-results-summary)
10. [Tech Stack](#10-tech-stack)
11. [Quickstart](#11-quickstart)
12. [Team](#12-team)

---

## 1. Project Overview

This project predicts **multiple movie genres simultaneously** from two independent data modalities:

| Modality | Input | Model |
|---|---|---|
| 📝 **Text** | Plot synopsis (NLP) | LSTM + GloVe embeddings |
| 🖼️ **Image** | Movie poster (Computer Vision) | EfficientNet-B3 (fine-tuned) |
| 🔀 **Multimodal** | Plot + Poster | Late-fusion MLP |

Each modality can also be used **independently** for genre inference, enabling a direct comparison of **text-only vs. image-only vs. fused** performance.

> **Key challenge:** Multi-label classification — a single movie can belong to *multiple* genres simultaneously (e.g., *Action + Comedy + Romance*).

---

## 2. Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MULTIMODAL MOVIE GENRE CLASSIFICATION                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
  │  1. DATA       │    │  2. EDA        │    │  3. BASELINE   │
  │  COLLECTION    │───▶│  & ANALYSIS    │───▶│  MODEL         │
  │  (TMDb API)    │    │                │    │  SELECTION     │
  └────────────────┘    └────────────────┘    └───────┬────────┘
                                                       │
                         ┌─────────────────────────────┴──────────────────────┐
                         │                                                      │
                  ┌──────▼──────┐                                      ┌───────▼──────┐
                  │ TEXT MODEL  │                                      │ IMAGE MODEL  │
                  │ (NLP Path)  │                                      │ (CV Path)    │
                  │             │                                      │              │
                  │  Preprocess │                                      │  Preprocess  │
                  │  → GloVe    │                                      │  → ImageNet  │
                  │  → LSTM     │                                      │  → EffNetB3  │
                  │  → Sigmoid  │                                      │  → Sigmoid   │
                  └──────┬──────┘                                      └───────┬──────┘
                         │  256-d text embedding                               │  256-d image embedding
                         └─────────────────────┬───────────────────────────────┘
                                               │
                                       ┌───────▼────────┐
                                       │  MULTIMODAL    │
                                       │  FUSION        │
                                       │  concat(512-d) │
                                       │  → MLP         │
                                       │  → Genres      │
                                       └────────────────┘
```

---

## 3. Repository Structure

```
Multimodal-Movie-Genre-Classification/
├── Data_Colllection_EDA.ipynb              # Stage 1 & 2: TMDb data collection + EDA
├── LSTM_Model.ipynb                        # Stage 3: NLP pipeline + LSTM training
├── EfficientNetModel with acccuracy        # Stage 4: Image pipeline + EfficientNet training
│   and loss.ipynb
├── datasets/
│   ├── undersampled_movie_metadata.csv     # Balanced dataset (random under-sampling)
│   └── custom_balanced_movie_metadata.csv  # Custom-balanced dataset
├── Results & Evaluation/
│   ├── Data/                               # EDA plots (genre frequency, co-occurrence…)
│   ├── Text/                               # LSTM training curves (F1-micro, F1-macro)
│   └── Image/                             # EfficientNet plots, confusion matrices
├── Movie Genre Classification Website/    # Front-end demo source
└── index.html                             # Live Genre Genie website entry point
```

---

## 4. Stage 1 — Data Collection

**Notebook:** `Data_Colllection_EDA.ipynb`

### Data Source

- **API:** [The Movie Database (TMDb)](https://www.themoviedb.org/documentation/api)
- **Scale:** Up to ~50,000 movies across years 1970–2025
- **Fields fetched:** `movie_id`, `title`, `overview` (plot synopsis), `genre_ids`, `poster_path`, `release_year`, `vote_average`

### Cleaning Steps

| Step | Action |
|---|---|
| Missing data | Dropped rows with no `overview` or broken `poster_path` |
| Genre mapping | Converted numeric genre IDs → human-readable labels via TMDb genre map |
| Encoding | One-hot encoded with `sklearn.MultiLabelBinarizer` (20 unique genre tags) |
| Deduplication | Removed duplicate titles |
| Class balancing | Random under-sampling across genre combinations |

### Final Datasets

| File | Rows | Notes |
|---|---|---|
| `undersampled_movie_metadata.csv` | ~20,530 | Balanced via random under-sampling |
| `custom_balanced_movie_metadata.csv` | Varies | Custom balance for model training |

> The 12 most common genres were retained for classification: **Drama, Comedy, Documentary, Romance, Action, Thriller, Horror, Crime, Music, Animation, TV Movie, Family**.  
> 8 niche genres (Adventure, Sci-Fi, Fantasy, Mystery, History, War, Western) were excluded due to severe class imbalance.

---

## 5. Stage 2 — Exploratory Data Analysis

**Notebook:** `Data_Colllection_EDA.ipynb`

### EDA Visualizations

<table>
<tr>
<td align="center" width="50%">
<strong>Genre Frequency Distribution</strong><br/>
<img src="Results%20%26%20Evaluation/Data/genre%20frequency.png" alt="Genre Frequency" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Detailed Genre Frequency</strong><br/>
<img src="Results%20%26%20Evaluation/Data/genre%20frequency%20detailed.png" alt="Genre Frequency Detailed" width="100%"/>
</td>
</tr>
<tr>
<td align="center" width="50%">
<strong>Normalized Genre Co-occurrence Matrix</strong><br/>
<img src="Results%20%26%20Evaluation/Data/norm%20genre%20co-oc%20matrix.png" alt="Genre Co-occurrence Matrix" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Top-10 Genre Combinations</strong><br/>
<img src="Results%20%26%20Evaluation/Data/top%2010%20genre%20comb.png" alt="Top 10 Genre Combinations" width="100%"/>
</td>
</tr>
<tr>
<td align="center" width="50%">
<strong>Before Under-sampling</strong><br/>
<img src="Results%20%26%20Evaluation/Data/undersampling.png" alt="Before Undersampling" width="100%"/>
</td>
<td align="center" width="50%">
<strong>After Under-sampling</strong><br/>
<img src="Results%20%26%20Evaluation/Data/post%20undersampling.png" alt="After Undersampling" width="100%"/>
</td>
</tr>
</table>

### Key EDA Findings

| Finding | Detail |
|---|---|
| Label distribution | Drama, Comedy, Thriller dominate; niche genres are severely under-represented |
| Multi-label cardinality | Most movies carry **2–3 genre labels** simultaneously |
| Genre correlations | Romance ↔ Drama (high co-occurrence); Horror ↔ Thriller (frequent pair) |
| Class imbalance | Addressed via random under-sampling before model training |
| Vocabulary | ~67,771 unique tokens across all plot synopses |

---

## 6. Stage 3 — Text-Only Model (NLP)

**Notebook:** `LSTM_Model.ipynb`

### 6.1 NLP Preprocessing Pipeline

Raw text undergoes a 7-step cleaning pipeline before being fed to the model:

```
Raw Synopsis
    │
    ▼
① Strip non-alphabetic characters  (re.sub)
    │
    ▼
② Lowercase all tokens
    │
    ▼
③ Tokenize                          (NLTK word_tokenize)
    │
    ▼
④ Remove English stop-words         (nltk.corpus.stopwords)
    │
    ▼
⑤ Lemmatize verbs                   (WordNetLemmatizer)
    │
    ▼
⑥ Drop tokens shorter than 3 chars
    │
    ▼
⑦ Pad / truncate to 300 tokens      (fixed sequence length)
    │
    ▼
Cleaned Token Sequence (len = 300)
```

### 6.2 Word Embeddings — GloVe

> **GloVe** (Global Vectors for Word Representation) encodes semantic relationships by factorizing word co-occurrence statistics from a large text corpus.

| Property | Value |
|---|---|
| Model | `glove-wiki-gigaword-100` |
| Dimensions | 100-d per token |
| Training corpus | Wikipedia + Gigaword (6B tokens) |
| Embedding matrix shape | `(vocab_size + 1, 100)` |
| OOV handling | Zero-vector for unknown tokens |
| Trainable | ❌ **Frozen** — weights are not updated during training |

The frozen embedding layer ensures the pre-trained semantic geometry (e.g., *king − man + woman ≈ queen*) is preserved while the LSTM learns genre-relevant patterns on top.

### 6.3 Baseline Text Models Compared

Multiple text classification approaches were evaluated before selecting LSTM as the final model:

| Model | Approach | Strength | Limitation |
|---|---|---|---|
| **Logistic Regression** | TF-IDF bag-of-words | Fast, interpretable | Ignores word order and context |
| **Naive Bayes** | TF-IDF probabilities | Very fast training | Strong independence assumption |
| **Simple RNN** | Sequential hidden state | Captures word order | Vanishing gradient on long sequences |
| **LSTM** ✅ | Gated memory cells | Long-range dependencies; preserves plot context | More parameters than RNN |
| **GRU** | Simplified gating | Faster than LSTM, comparable accuracy | Slightly less expressive than LSTM |

**LSTM was selected** as the final text model because plot synopses are long sequences (up to 300 tokens) where genre-relevant context (e.g., a climactic twist mentioned late) must be retained from early tokens — a task where gated memory cells excel.

### 6.4 Final Architecture — LSTM

```
┌─────────────────────────────────────────────────────────┐
│                   LSTM Text Classifier                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input tokens  (batch × 300)                            │
│       │                                                  │
│       ▼                                                  │
│  Embedding Layer                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  GloVe glove-wiki-gigaword-100  (frozen)        │    │
│  │  Shape: (vocab_size+1, 100)                     │    │
│  └─────────────────────────────────────────────────┘    │
│       │  (batch × 300 × 100)                            │
│       ▼                                                  │
│  LSTM Layer                                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │  hidden_dim  = 256                              │    │
│  │  num_layers  = 1                                │    │
│  │  batch_first = True                             │    │
│  │  Forget gate / Input gate / Output gate         │    │
│  └─────────────────────────────────────────────────┘    │
│       │  Final hidden state  (batch × 256)              │
│       ▼                                                  │
│  Linear Layer   256 → 20                                │
│       │                                                  │
│       ▼                                                  │
│  Sigmoid (per-genre probability)                         │
│       │                                                  │
│       ▼                                                  │
│  Genre predictions  (batch × 20)                        │
└─────────────────────────────────────────────────────────┘

  Loss:       BCEWithLogitsLoss (binary cross-entropy per label)
  Optimizer:  Adam  |  lr = 0.0001
  Batch size: 64    |  Epochs: 10   |  Train/Val: 80/20
```

### 6.5 Text-Only Evaluation

Training curves saved under `Results & Evaluation/Text/`:

<table>
<tr>
<td align="center" width="50%">
<strong>Train vs Validation F1-Micro</strong><br/>
<img src="Results%20%26%20Evaluation/Text/T%20vs%20V%20F1%20micro.png" alt="F1 Micro" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Train vs Validation F1-Macro</strong><br/>
<img src="Results%20%26%20Evaluation/Text/T%20vs%20V%20F1%20macro.png" alt="F1 Macro" width="100%"/>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<strong>Overfitting Accuracy Check</strong><br/>
<img src="Results%20%26%20Evaluation/Text/overfit%20acc.png" alt="Overfit Accuracy" width="50%"/>
</td>
</tr>
</table>

| Metric | Description |
|---|---|
| **F1-micro** | Aggregates TP/FP/FN across all labels — favours frequent genres |
| **F1-macro** | Unweighted mean F1 across all 20 genres — penalises poor rare-genre recall |

**Standalone Text-Only Prediction:** A `predict_movie(model, device, val_loader)` helper processes any raw plot synopsis through the full NLP pipeline and returns per-genre probability scores — enabling **text-only genre inference** without any poster image.

---

## 7. Stage 4 — Image-Only Model (CNN)

**Notebook:** `EfficientNetModel with acccuracy and loss.ipynb`

### 7.1 Poster Preprocessing

```
Raw Poster URL
    │
    ▼
Download from TMDb CDN
    │
    ▼
① Resize to 256 × 256
    │
    ▼
② Random crop → 224 × 224          [train only]
    │
    ▼
③ Random horizontal flip (p=0.5)   [train only]
    │
    ▼
④ Color jitter                     [train only]
   brightness/contrast/saturation ± 0.1
    │
    ▼
⑤ Normalize
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]  (ImageNet statistics)
    │
    ▼
Tensor  (3 × 224 × 224)
```

### 7.2 Baseline Image Models Compared

Several CNN architectures were evaluated to select the best feature extractor for movie poster classification:

| Model | Params | ImageNet Top-1 | Val Accuracy (posters) | Notes |
|---|---|---|---|---|
| **ResNet-50** | 25 M | 76.1 % | ~79 % | Strong baseline; deeper variants prone to over-fitting on small poster sets |
| **ResNet-101** | 44 M | 77.4 % | ~80 % | Marginal gain over ResNet-50 at higher cost |
| **VGG-16** | 138 M | 71.6 % | ~76 % | Very large; slow training |
| **MobileNet-V2** | 3.4 M | 71.8 % | ~74 % | Fast but accuracy too low for poster nuances |
| **EfficientNet-B3** ✅ | 12 M | 81.6 % | **~82 %** | Best accuracy-to-parameter ratio; compound scaling balances depth, width, resolution |

<table>
<tr>
<td align="center" width="50%">
<strong>ResNet Baseline — Training Output</strong><br/>
<img src="Results%20%26%20Evaluation/Image/RESNET_MODEL_OUTPUT.jpg" alt="ResNet Model Output" width="100%"/>
</td>
<td align="center" width="50%">
<strong>ResNet Baseline — Classification Report</strong><br/>
<img src="Results%20%26%20Evaluation/Image/Classification%20Report_RESNET.jpg" alt="ResNet Classification Report" width="100%"/>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<strong>Baseline Model Accuracy &amp; Loss Comparison</strong><br/>
<img src="Results%20%26%20Evaluation/Image/baseline%20plots.jpg" alt="Baseline Plots" width="70%"/>
</td>
</tr>
</table>

**EfficientNet-B3 was selected** because its compound scaling law (jointly scaling network depth, width, and input resolution) yields the highest accuracy-per-parameter ratio — critical for poster images where subtle visual cues (colour palette, typography, lighting) differentiate genres.

### 7.3 Final Architecture — EfficientNet-B3

```
┌──────────────────────────────────────────────────────────┐
│              EfficientNet-B3 Image Classifier             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Input poster  (batch × 3 × 224 × 224)                  │
│       │                                                   │
│       ▼                                                   │
│  EfficientNet-B3 Backbone                                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Pretrained: ImageNet (IMAGENET1K_V1)            │    │
│  │  Fine-tuned: ✅ entire backbone trainable        │    │
│  │                                                  │    │
│  │  MBConv blocks with Squeeze-and-Excitation       │    │
│  │  Compound scaling: depth=1.4 / width=1.2 /      │    │
│  │                    resolution=300                │    │
│  └──────────────────────────────────────────────────┘    │
│       │  Feature maps                                     │
│       ▼                                                   │
│  Global Average Pooling  →  1536-d feature vector        │
│       │                                                   │
│       ▼                                                   │
│  Dropout  (p = 0.3)                                      │
│       │                                                   │
│       ▼                                                   │
│  Linear  1536 → num_genres                               │
│       │                                                   │
│       ▼                                                   │
│  Sigmoid  (multi-label)  /  Softmax  (single-label)      │
│       │                                                   │
│       ▼                                                   │
│  Genre predictions  (batch × num_genres)                 │
└──────────────────────────────────────────────────────────┘

  Mixed precision:  torch.amp.GradScaler + autocast
  Batch size: 32   |   Epochs: 20–30
```

### 7.4 Classification Modes

| Mode | Output activation | Loss function | Use case |
|---|---|---|---|
| **Multi-label** | Sigmoid (per genre) | `BCEWithLogitsLoss` with `pos_weight` | Multiple genres per poster |
| **Single-label** | Softmax | `CrossEntropyLoss` (LabelEncoder) | Primary genre only |

### 7.5 Image-Only Evaluation

<table>
<tr>
<td align="center" width="50%">
<strong>Accuracy &amp; Loss (EfficientNet with embeddings)</strong><br/>
<img src="Results%20%26%20Evaluation/Image/with%20embed%20EffNet.jpg" alt="EfficientNet with Embeddings" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Accuracy &amp; Loss (Baseline EfficientNet)</strong><br/>
<img src="Results%20%26%20Evaluation/Image/acc%20and%20loss%20plots.jpg" alt="Accuracy and Loss" width="100%"/>
</td>
</tr>
<tr>
<td align="center" width="50%">
<strong>Validation Metrics per Epoch</strong><br/>
<img src="Results%20%26%20Evaluation/Image/validation%20metric.png" alt="Validation Metrics" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Genre-wise Performance Metrics</strong><br/>
<img src="Results%20%26%20Evaluation/Image/genre%20wise%20preformance%20metrics.png" alt="Genre Wise Performance" width="100%"/>
</td>
</tr>
<tr>
<td align="center" width="50%">
<strong>Confusion Matrix — Multi-label</strong><br/>
<img src="Results%20%26%20Evaluation/Image/CM_multi-genre_image.jpg" alt="Confusion Matrix Multi-label" width="100%"/>
</td>
<td align="center" width="50%">
<strong>Confusion Matrix — Single-label</strong><br/>
<img src="Results%20%26%20Evaluation/Image/CM_single_genre_image.jpg" alt="Confusion Matrix Single-label" width="100%"/>
</td>
</tr>
</table>

**Sample movie posters** from the training set:

<div align="center">
<img src="Results%20%26%20Evaluation/Image/sample_posters.jpg" alt="Sample Movie Posters" width="70%"/>
</div>

| Epoch | Train Loss | Val Loss | Train Accuracy | Val Accuracy |
|---|---|---|---|---|
| 1 | 1.0211 | 0.9490 | 0.7693 | 0.7817 |
| 5 | 0.8417 | 0.9219 | 0.8085 | 0.8020 |
| 10 | 0.6119 | 1.0581 | 0.8581 | 0.8030 |
| 15 | 0.3607 | 1.4683 | 0.9180 | 0.8159 |
| **20** | **0.2706** | **1.6289** | **0.9387** | **0.8218** |

**Standalone Image-Only Prediction:** The trained EfficientNet-B3 classification head can predict genres from any poster image independently — **image-only genre inference** without needing a plot synopsis.

---

## 8. Stage 5 — Multimodal Fusion

The fusion architecture combines learned representations from both modalities via **late fusion** — each modality is encoded independently by its specialist encoder, and their embeddings are concatenated before a final shared classification head.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     MULTIMODAL FUSION ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Plot Synopsis                        Movie Poster                     │
│       │                                     │                          │
│       ▼                                     ▼                          │
│  NLP Pipeline                         Image Pipeline                   │
│  (tokenize, GloVe)                    (resize, normalize)              │
│       │                                     │                          │
│       ▼                                     ▼                          │
│  LSTM Encoder                         EfficientNet-B3                  │
│  (1-layer, 256 hidden)                (fine-tuned backbone)            │
│       │                                     │                          │
│       ▼                                     ▼                          │
│  Final hidden state                   Global Avg Pool → Linear         │
│  256-d text embedding                 256-d image embedding            │
│       │                                     │                          │
│       └──────────────┬──────────────────────┘                         │
│                      │ torch.cat([text_emb, img_emb])                  │
│                      ▼                                                  │
│              512-d joint representation                                 │
│                      │                                                  │
│                      ▼                                                  │
│              MLP Classification Head                                    │
│              ┌────────────────────────────┐                            │
│              │  Linear(512 → 256)         │                            │
│              │  ReLU                      │                            │
│              │  Dropout                   │                            │
│              │  Linear(256 → num_genres)  │                            │
│              │  Sigmoid                   │                            │
│              └────────────────────────────┘                            │
│                      │                                                  │
│                      ▼                                                  │
│              Multi-label Genre Predictions                              │
└────────────────────────────────────────────────────────────────────────┘
```

### Why Late Fusion Works Here

| Signal | What it captures | Example |
|---|---|---|
| **Text (plot synopsis)** | Narrative semantics, character dynamics, plot themes | "A group of friends are stalked by a killer" → Horror |
| **Image (movie poster)** | Visual aesthetics, colour palette, typography style | Dark desaturated tones, silhouettes → Thriller / Horror |
| **Combined** | Complementary cues — posters can mislead; text can be vague | Romantic comedy with action-style poster |

---

## 9. Results Summary

| Model | Modality | Task | Val Accuracy | Notes |
|---|---|---|---|---|
| Logistic Regression | Text (TF-IDF) | Multi-label | Baseline | Used as lower-bound reference |
| Simple RNN | Text | Multi-label | — | Struggled with long synopses |
| **LSTM + GloVe** | **Text only** | **Multi-label** | **F1-micro / F1-macro tracked** | Frozen GloVe, 10 epochs |
| ResNet-50 | Image only | Multi-label | ~79 % | Evaluated as baseline |
| **EfficientNet-B3** | **Image only** | **Multi-label** | **82.18 %** | Fine-tuned, 20 epochs |
| EfficientNet-B3 | Image only | Single-label | 28.29 % (val) | Severe overfitting (train 98.16 %) |
| **LSTM + EfficientNet** | **Text + Image** | **Multi-label** | **Best combined F1** | 512-d late fusion MLP |

All training curves, confusion matrices, and per-genre classification reports are in the `Results & Evaluation/` folder.

---

## 10. Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Data collection** | `requests`, `tqdm`, TMDb REST API |
| **Data processing** | `pandas`, `numpy`, `sklearn` (`MultiLabelBinarizer`, `LabelEncoder`) |
| **EDA & visualisation** | `matplotlib`, `seaborn` |
| **NLP** | `nltk` (tokenise, stop-words, lemmatise), `re`, `gensim` (GloVe), `keras` (padding) |
| **Deep learning** | `PyTorch`, `torchvision` |
| **Evaluation** | `sklearn` (F1-score, confusion matrix, classification report) |
| **Training acceleration** | `torch.amp` (mixed-precision), CUDA GPU |
| **Deployment** | Netlify (live demo), HTML/CSS/JS front-end |

---

## 11. Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/patilanupam/Multimodal-Movie-Genre-Classification.git
cd Multimodal-Movie-Genre-Classification

# 2. Open the desired notebook in Google Colab or Jupyter
#    ├── Data_Colllection_EDA.ipynb                           → Stage 1 & 2
#    ├── LSTM_Model.ipynb                                     → Stage 3 (text model)
#    └── "EfficientNetModel with acccuracy and loss.ipynb"    → Stage 4 (image model)

# 3. Mount Google Drive and update the path variables at the top of each notebook
#    embedding_path   → path to glove-wiki-gigaword-100 vectors
#    local_poster_dir → path to downloaded TMDb poster images
#    csv_path         → path to undersampled_movie_metadata.csv
```

> **External assets (not in this repo):**
> - Pre-trained GloVe vectors (`glove-wiki-gigaword-100`) — download via `gensim.downloader`
> - TMDb poster images — downloaded on-the-fly during `Data_Colllection_EDA.ipynb`
>
> Update `embedding_path` and `local_poster_dir` variables at the top of each notebook.

---

## 12. Team

| Name | Role | LinkedIn |
|---|---|---|
| **Anupam Patil** | Data Specialist & Image Modelling Lead | [linkedin.com/in/anupam-patil](https://www.linkedin.com/in/anupam-patil) |
| **Nandini Soni** | Text Model & Fusion Lead | [linkedin.com/in/nandini-soni-901bb580](https://www.linkedin.com/in/nandini-soni-901bb580/) |
