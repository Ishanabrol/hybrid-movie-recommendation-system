# 🎬 Movie Recommendation System

An end-to-end movie recommendation system built on the MovieLens 1M dataset,
implementing and comparing four recommendation approaches — Collaborative 
Filtering, Content Based Filtering, Neural Collaborative Filtering and a 
Hybrid Model — deployed as an interactive Streamlit web application.

---

## 📦 Dataset

**MovieLens 1M** — [grouplens.org](https://grouplens.org/datasets/movielens/1m/)

- 1,000,209 ratings across 6,040 users and 3,883 movies
- Rating scale: 1–5 stars
- Rating period: April 2000 — February 2003
- Matrix sparsity: 95.53%

---

## 🤖 Models Implemented

### 1. Collaborative Filtering (SVD)
Matrix factorization compressing the sparse user-movie matrix into latent 
factor representations. Tuned via GridSearchCV with 3-fold cross validation.
- **RMSE: 0.9372 | MAE: 0.7440**

### 2. Content Based Filtering
Genre-based recommendations using TF-IDF vectorization and cosine similarity 
across 18 unique genre tokens and a 3,883 × 3,883 similarity matrix.
- **P@10: 0.0153 | NDCG@10: 0.0186**

### 3. Neural Collaborative Filtering (NCF)
Deep learning model replacing SVD's dot product with neural layers to capture 
non-linear user-movie interaction patterns. Uses early stopping and 
ReduceLROnPlateau to prevent overfitting.
- Architecture: Embeddings(64) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Sigmoid
- 650,625 trainable parameters
- **RMSE: 0.9587 | MAE: 0.7598**

### 4. Hybrid Model (Tuned)
Weighted combination of all three models. MinMaxScaler normalizes all scores 
to 0–1 before combining to fix scale mismatch. Weights tuned based on 
evaluation results.
- Tuned weights: SVD=0.35, NCF=0.50, Content=0.15
- **P@10: 0.1041 | NDCG@10: 0.1018**

---

## 📊 Results

### Rating Prediction Accuracy (lower is better)

| Model | RMSE | MAE |
|-------|------|-----|
| SVD | 0.9372 | 0.7440 |
| Neural CF | 0.9587 | 0.7598 |

### Recommendation Quality @ K=10 (higher is better)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|-------------|-----------|---------|
| SVD | 0.0837 | 0.0167 | 0.0747 |
| Neural CF | 0.0980 | 0.0208 | 0.0970 |
| Content Based | 0.0153 | 0.0032 | 0.0186 |
| **Hybrid (Tuned)** | **0.1041** | **0.0277** | **0.1018** |

**Key findings:**
- Hybrid outperforms all individual models on every ranking metric
- NCF beats SVD on ranking metrics despite SVD having better RMSE —
  rating accuracy does not equal recommendation quality
- Weight tuning improved hybrid average score from 0.0750 to 0.0779

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| Environment | Anaconda, Jupyter Notebook |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Collaborative Filtering | scikit-surprise (SVD) |
| Content Based | scikit-learn (TF-IDF, cosine similarity) |
| Deep Learning | TensorFlow 2.20, Keras |
| Deployment | Streamlit 1.50 |

---

## ⚠️ Known Limitations

- Dataset contains movies up to 2003 only — no modern movies
- Cold start problem — new users cannot get collaborative filtering recommendations
- Hybrid model intersection reduces movie pool from 3,706 to ~1,857
- Content based uses genres only — no plot, cast or director features
- Models are static — no real time updates with new ratings
- No diversity optimization — models optimize for predicted rating only

---

## 📈 Future Improvements

- Union approach with confidence weighting to expand hybrid movie pool
- Richer content features — plot descriptions, cast, director
- Popularity based fallback for cold start users
- Post-processing re-ranking for recommendation diversity
- Real time model updates with new rating data
- GPU training for larger Neural CF architecture

---

## 👤 Author

**Ishan**
Data Science Portfolio Project
Dataset: MovieLens 1M — GroupLens Research
