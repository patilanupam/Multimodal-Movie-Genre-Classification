# Multimodal Movie Genre Classification

> **Course:** EAS 510LEC (AI Basics) – Spring 2025  
> **Author:** Anupam Patil (Data Specialist & Image Modelling Lead)  
> **Co-Author:** Nandini Soni (Text Model & Fusion Lead)

🌐 **Live Demo:** [https://genregenie.netlify.app/](https://genregenie.netlify.app/)

---

## Overview

This project classifies movie genres from two independent modalities — **plot summaries (text)** and **movie posters (images)** — and fuses them into a **multimodal classifier**. The pipeline follows five stages:

```
Data Collection → EDA → Text Model (LSTM) → Image Model (EfficientNet) → Multimodal Fusion
```

Each modality can also be used **independently** for genre prediction, allowing a direct comparison of text-only vs. image-only vs. fused performance.

---

## Repository Structure

| File / Folder | Description |
|---|---|
| `Data_Colllection_EDA.ipynb` | TMDb data collection, cleaning, and exploratory data analysis |
| `LSTM_Model.ipynb` | NLP pipeline: text preprocessing, GloVe embeddings, LSTM model |
| `EfficientNetModel with acccuracy and loss.ipynb` | Image pipeline: poster preprocessing, EfficientNet-B3 model |
| `datasets/` | Balanced CSV files used for training (`undersampled_movie_metadata.csv`, `custom_balanced_movie_metadata.csv`) |
| `Results & Evaluation/` | Training plots, confusion matrices, and classification reports |
| `index.html` | Source for the live Genre Genie website |

---

## Stage 1 — Data Collection & Cleaning

**Notebook:** `Data_Colllection_EDA.ipynb`

- **Source:** The Movie Database (TMDb) API — fetched up to 50,000 movies (1970–2025) including title, plot overview, genre IDs, and poster URL.
- **Columns kept:** `movie_id`, `title`, `genres`, `overview`, `year`, `vote_avg`
- **Cleaning steps:**
  - Dropped rows with missing `overview` or broken `poster_path`
  - Parsed genre IDs to human-readable genre names using the TMDb genre map
  - One-hot encoded genres with `sklearn.MultiLabelBinarizer` (20 unique genre tags)
  - Removed duplicate titles
- **Class balancing:** Applied random under-sampling across genre combinations to reduce genre imbalance; balanced dataset saved to `datasets/undersampled_movie_metadata.csv`.

---

## Stage 2 — Exploratory Data Analysis (EDA)

**Notebook:** `Data_Colllection_EDA.ipynb` / `LSTM_Model.ipynb`

| Analysis | Visualization |
|---|---|
| Genre frequency distribution | Bar chart with normal-curve overlay |
| Log-transformed genre-pair frequencies | Long-tail distribution plot |
| Normalized genre co-occurrence matrix | Heatmap (correlation −1 → 1) |
| Top-10 genre combinations | Ranked bar chart |
| Before/after under-sampling comparison | Side-by-side genre count plots |
| Synopsis word-count distribution | Histogram |
| Number of genres per movie (multi-label cardinality) | Count plot |

Key findings: Drama, Comedy, and Thriller dominate; most movies carry 2–3 genre labels; rare genres (Documentary, Foreign) are heavily under-represented before balancing.

**EDA plots** are saved under `Results & Evaluation/Data/`.

---

## Stage 3 — Text-Only Model (LSTM + GloVe)

**Notebook:** `LSTM_Model.ipynb`

### 3.1 Text Preprocessing (NLP Pipeline)

1. Strip non-alphabetic characters with `re.sub`
2. Lowercase
3. Tokenise with NLTK `word_tokenize`
4. Remove English stop-words (`nltk.corpus.stopwords`)
5. Lemmatise verbs with `WordNetLemmatizer`
6. Drop tokens shorter than 3 characters
7. Pad/truncate sequences to a fixed length of **300 tokens**

### 3.2 Word Embeddings — GloVe

- **Model:** `glove-wiki-gigaword-100` (100-dimensional, trained on Wikipedia + Gigaword)
- An **embedding matrix** of shape `(vocab_size + 1, 100)` is constructed by looking up each vocabulary token in the GloVe index; out-of-vocabulary tokens remain as zero vectors.
- The embedding layer is **frozen** during training (weights are not updated).

### 3.3 Model Architecture

```
Input (batch, 300)  →  Embedding Layer (frozen GloVe, 100-d)
                    →  LSTM (hidden_dim = 256, num_layers = 1, batch_first = True)
                    →  Final hidden state (256-d)
                    →  Linear (256 → 20)
                    →  Sigmoid activation per genre
```

- **Task:** Multi-label classification (each movie can belong to multiple genres simultaneously)
- **Loss:** `BCEWithLogitsLoss` (binary cross-entropy applied per label)
- **Optimizer:** Adam, `lr = 0.0001`
- **Batch size:** 64 | **Epochs:** 10
- **Train/Val split:** 80 / 20

### 3.4 Evaluation Metrics

- Accuracy, **F1-micro**, **F1-macro** tracked per epoch on both train and validation sets
- Plots saved under `Results & Evaluation/Text/`

| Metric | Description |
|---|---|
| F1-micro | Aggregates TP/FP/FN across all labels — favours frequent genres |
| F1-macro | Unweighted mean F1 across all 20 genres — penalises poor rare-genre recall |

### 3.5 Standalone Text-Only Prediction

A `predict_movie(model, device, val_loader)` helper feeds a raw plot synopsis through the full NLP pipeline and returns per-genre probability scores, enabling **text-only genre inference** without any image input.

---

## Stage 4 — Image-Only Model (EfficientNet-B3)

**Notebook:** `EfficientNetModel with acccuracy and loss.ipynb`

### 4.1 Poster Preprocessing

| Step | Detail |
|---|---|
| Resize | 256 × 256 |
| Random crop (train) | 224 × 224 |
| Random horizontal flip (train) | p = 0.5 |
| Color jitter (train) | brightness / contrast / saturation ± 0.1 |
| Normalize | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |

### 4.2 Model Architecture — EfficientNet-B3

```
Input (batch, 3, 224, 224)
    →  EfficientNet-B3 backbone (pretrained on ImageNet)
    →  Global Average Pooling  (1536-d feature vector)
    →  Dropout (p = 0.3)
    →  Linear (1536 → num_genres)
    →  Sigmoid / Softmax
```

- **Backbone:** `torchvision.models.efficientnet_b3` with `EfficientNet_B3_Weights.IMAGENET1K_V1`
- The entire backbone is **fine-tuned** (not frozen) to adapt ImageNet features to movie poster aesthetics.
- Mixed-precision training via `torch.amp.GradScaler` + `autocast` for faster GPU computation.

### 4.3 Two Classification Modes

| Mode | Output | Loss | Use case |
|---|---|---|---|
| **Multi-label** | Sigmoid over 20 genres | BCEWithLogitsLoss | Multiple genres per poster |
| **Single-label** | Softmax over N genres | CrossEntropyLoss (LabelEncoder) | Primary genre only |

- **Batch size:** 32 | **Epochs:** 20–30
- Confusion matrices and genre-wise precision/recall/F1 saved under `Results & Evaluation/Image/`

### 4.4 Standalone Image-Only Prediction

The trained EfficientNet-B3 head can classify any poster image independently, making **image-only genre inference** possible without a plot synopsis.

---

## Stage 5 — Multimodal Fusion

The fusion model concatenates the learned representations from both modalities before a final classification head:

```
Plot Synopsis  →  LSTM encoder  →  256-d text embedding   ─┐
                                                             ├─ concat (512-d)  →  MLP  →  genres
Poster Image   →  EfficientNet  →  256-d image embedding  ─┘
```

- The LSTM produces a **256-d** sequence summary (final hidden state).
- EfficientNet's pooled feature map is projected to **256-d**.
- The concatenated **512-d** vector is passed through a small MLP with sigmoid outputs for final multi-label classification.
- This allows the model to leverage **complementary signals**: text captures plot semantics; images capture visual genre cues (e.g., dark palettes for Horror, vibrant colours for Comedy).

---

## Results Summary

| Model | Modality | Task | Key Metric |
|---|---|---|---|
| LSTM + GloVe | Text only | Multi-label | F1-macro / F1-micro tracked over 10 epochs |
| EfficientNet-B3 | Image only | Multi-label | Accuracy & loss over 20–30 epochs |
| EfficientNet-B3 | Image only | Single-label | Classification report + confusion matrix |
| LSTM + EfficientNet MLP | Text + Image | Multi-label | Combined F1 (fusion) |

All training curves, confusion matrices, and per-genre classification reports are in the `Results & Evaluation/` folder.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| NLP | `nltk`, `re`, `keras` (padding), `gensim` (GloVe) |
| Deep Learning | `PyTorch`, `torchvision` |
| Evaluation | `sklearn` (F1, confusion matrix, MultiLabelBinarizer) |
| Data Source | TMDb API (`requests`, `tqdm`) |

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/patilanupam/Multimodal-Movie-Genre-Classification.git

# 2. Open any notebook in Google Colab or Jupyter
#    - Data_Colllection_EDA.ipynb          → data & EDA
#    - LSTM_Model.ipynb                    → text-only model
#    - EfficientNetModel with acccuracy and loss.ipynb  → image-only model

# 3. Mount Google Drive and update dataset/embedding paths at the top of each notebook
```

> **Note:** Pre-trained GloVe vectors (`glove-wiki-gigaword-100`) and downloaded poster images are stored in Google Drive and are not committed to the repository. Update the `embedding_path` and `local_poster_dir` variables accordingly.

---

## Team

| Name | Role | LinkedIn |
|---|---|---|
| Anupam Patil | Data Specialist & Image Modelling Lead | [linkedin.com/in/anupam-patil](https://www.linkedin.com/in/anupam-patil) |
| Nandini Soni | Text Model & Fusion Lead | [linkedin.com/in/nandini-soni-901bb580](https://www.linkedin.com/in/nandini-soni-901bb580/) |
