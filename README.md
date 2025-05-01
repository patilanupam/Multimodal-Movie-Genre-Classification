# Multimodal-Movie-Genre
DATA->EDA->TEXT MODEL-> IMAGE MODEL-> FUSION(TEXT+IMAGE)
Here’s a down-to-earth README that walks through exactly what I’ve done so far—just like I’d explain it to a classmate.

---

# Multimodal Movie Genre Classification

> **Course:** EAS 510LEC (AI Basics) – Spring 2025  
> **Author:** Anupam Patil (Data Specialist role and Image Modelling lead)  
**Co-Author:** Nandini Soni (Text Model and Fusion Lead) 
This repo is my semester project for classifying movie genres using **both** plot summaries and poster images. I’ve broken my work into three main notebooks:

- **Data_Colllection_EDA.ipynb** – grabbing the data, cleaning it, and running some EDA  
- **LSTM_Model.ipynb** – building a text-only model with an LSTM + GloVe  
- **EfficientNetModel with acccuracy and loss.ipynb** – training an image-only classifier  

---

## 1. Data Collection & Cleaning

- **TMDb API**: Pulled metadata (title, overview, genre IDs, poster URLs) for movies from 1970–2025.  
- **Filtering**: Kept only entries with a valid `overview` and working `poster_path`.  
- **Balanced CSV**: Used under-sampling to smooth out genre imbalance; saved as `datasets/undersample_movies.csv`.  
- **Mount & Setup**: In Colab I mounted Google Drive, set up paths, and loaded all CSVs/images.

---

## 2. Exploratory Data Analysis (EDA)

1. **Basic stats**:  
   - Total movies  
   - Missing values per column  
2. **Genre breakdown**:  
   - Bar plots of genre frequencies (with normal-curve overlay)  
   - Log-transformed view of rare vs. common genre pairs  
3. **Text lengths**:  
   - Distribution of synopsis lengths (word counts)  
4. **Multi-genre counts**:  
   - How many movies have 1, 2, 3… genres  
5. **Metadata sanity checks**:  
   - Check for weird JSON in genre fields  
   - Spot any poster URL problems before training  

*(All of this lives in `Data_Colllection_EDA.ipynb`.)*

---

## 3. Text-Only Model (LSTM)

- **Preprocessing**  
  - Parsed the JSON genre list into Python lists  
  - Lowercased, removed stopwords, tokenized, and padded synopses  
- **Embeddings**  
  - Loaded pre-trained GloVe vectors  
  - Built an embedding matrix for our vocabulary  
- **Model**  
  - Single-layer LSTM → fully-connected → sigmoid for each genre  
  - Multi-label setup (binary cross-entropy loss)  
- **Training**  
  - Trained for ~10 epochs on our train split  
  - Monitored train vs. validation loss and F1-score  
- **Prediction fn**  
  - Simple helper to feed in a new plot and get genre probs  

*(See `LSTM_Model.ipynb` for code + plots.)*

---

## 4. Image-Only Model (EfficientNet)

- **Preprocessing**  
  - Resized posters to 224×224, normalized to ImageNet stats  
  - Added random crops & horizontal flips during training  
- **Model**  
  - Used `torchvision`’s EfficientNet-B0 (pretrained)  
  - Replaced the final head with a single-layer classifier  
- **Training**  
  - Trained for ~8 epochs  
  - Logged accuracy & loss curves  
- **Results**  
  - Reached ~68% accuracy on my validation set  

*(Details in `EfficientNetModel with acccuracy and loss.ipynb`.)*

---

## 5. Where I’m Heading Next

- **Fusion model**: concatenate LSTM text embeddings + EfficientNet image embeddings, then train a small MLP on top  
- **Compare**: text-only vs. image-only vs. fused performance (accuracy, precision, recall, F1)  
- **Multi-label for images**: move from single-label EfficientNet to true multi-label output  
- **Bonus**: simple Flask app to upload a poster & plot and see genre predictions  

---
