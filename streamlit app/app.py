
# ============================================================
# MOVIE RECOMMENDATION SYSTEM — STREAMLIT APP
# ============================================================
# To run this app:
# 1. Open Anaconda Prompt
# 2. conda activate recsys
# 3. cd "OneDrive/Documents/Recommendation System"
# 4. streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from surprise import SVD
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# PAGE CONFIGURATION
# Must be the FIRST streamlit command in the script
# ============================================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    .movie-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #4f8bf9;
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .genre-badge {
        background-color: #4f8bf9;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 4px;
        display: inline-block;
    }
    
    .score-display {
        color: #4f8bf9;
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .rank-number {
        color: #4f8bf9;
        font-size: 28px;
        font-weight: 800;
        opacity: 0.5;
        float: right;
    }
    
    .section-header {
        color: #4f8bf9;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4f8bf9;
    }
    
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS AND PATHS
# ============================================================
BASE_PATH = r"C:\Users\ishaa\OneDrive\Documents\Recommendation System"
MODELS_PATH = f"{BASE_PATH}/models"

# ============================================================
# MODEL LOADING WITH CACHING
# @st.cache_resource loads models once and keeps in memory
# Without caching models reload on every interaction — very slow
# ============================================================
@st.cache_resource
def load_all_models():
    """Load all trained models and artifacts — cached on first load."""
    
    # Load SVD model saved in Phase 3
    with open(f'{MODELS_PATH}/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    
    # Load NCF model and encoders saved in Phase 5
    ncf_model = keras.models.load_model(f'{MODELS_PATH}/ncf_model.keras')
    with open(f'{MODELS_PATH}/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open(f'{MODELS_PATH}/movie_encoder.pkl', 'rb') as f:
        movie_encoder = pickle.load(f)
    
    # Load content based models saved in Phase 4
    cosine_sim = np.load(f'{MODELS_PATH}/cosine_sim.npy')
    movie_indices = pd.read_pickle(f'{MODELS_PATH}/movie_indices.pkl')
    
    # Load hybrid weights config saved in Phase 6 and tuned in Phase 7
    with open(f'{BASE_PATH}/hybrid_config.json', 'r') as f:
        hybrid_config = json.load(f)
    
    return (svd_model, ncf_model, user_encoder, movie_encoder,
            cosine_sim, movie_indices, hybrid_config)


@st.cache_data
def load_data():
    """
    Load processed data files — cached on first load.
    @st.cache_data used for dataframes and serializable objects.
    @st.cache_resource used for models and non-serializable objects.
    """
    train = pd.read_csv(f'{BASE_PATH}/train.csv')
    movies = pd.read_csv(f'{BASE_PATH}/movies.csv')
    
    # Apply same genre preprocessing as all previous notebooks
    # Must match exactly what was used to build cosine similarity matrix
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['genres_clean'] = movies['genres_clean'].str.replace('Sci-Fi', 'SciFi', regex=False)
    movies['genres_clean'] = movies['genres_clean'].str.replace('Film-Noir', 'FilmNoir', regex=False)
    movies['genres_clean'] = movies['genres_clean'].str.replace("Children's", 'Childrens', regex=False)
    movies['genres_clean'] = movies['genres_clean'].fillna('')
    
    # Load evaluation results if available — used in performance table
    eval_path = f'{BASE_PATH}/final_evaluation_results.csv'
    eval_results = pd.read_csv(eval_path) if os.path.exists(eval_path) else None
    
    return train, movies, eval_results


# ============================================================
# RECOMMENDATION FUNCTIONS
# Same scoring functions used throughout all notebooks
# ============================================================
def get_svd_scores(user_id, svd_model, movies_df, rated_movie_ids):
    """
    Get SVD predicted ratings for all unrated movies.
    SVD inference is fast — just a dot product of learned vectors.
    Returns dict: {movie_id: predicted_rating}
    """
    all_movie_ids = movies_df['movie_id'].unique()
    
    # Only predict for movies user hasn't seen
    unrated_movies = [m for m in all_movie_ids if m not in rated_movie_ids]
    
    # .est is Surprise's estimated rating for the (user, movie) pair
    scores = {
        movie_id: svd_model.predict(user_id, movie_id).est
        for movie_id in unrated_movies
    }
    return scores


def get_ncf_scores(user_id, ncf_model, movies_df, rated_movie_ids,
                   user_encoder, movie_encoder):
    """
    Get NCF predicted ratings for all unrated movies.
    Uses batch prediction for efficiency.
    Returns dict: {movie_id: predicted_rating}
    """
    # Cannot predict for users not seen during training
    if user_id not in user_encoder.classes_:
        return {}
    
    # Encode user ID to integer index used during training
    user_idx = user_encoder.transform([user_id])[0]
    
    all_movie_ids = movies_df['movie_id'].unique()
    
    # Filter to unrated movies that exist in encoder
    unrated_movies = [
        m for m in all_movie_ids
        if m not in rated_movie_ids
        and m in movie_encoder.classes_
    ]
    
    # Encode all movie IDs to integer indices
    movie_indices_arr = movie_encoder.transform(unrated_movies)
    
    # Repeat user index for every movie — model needs (user, movie) pairs
    user_arr = np.array([user_idx] * len(unrated_movies))
    movie_arr = np.array(movie_indices_arr)
    
    # Batch prediction — all movies scored in one efficient call
    predictions = ncf_model.predict(
        [user_arr, movie_arr],
        batch_size=512,
        verbose=0
    ).flatten()
    
    # Reverse normalization from 0-1 back to 1-5 scale
    predictions = predictions * 4 + 1
    predictions = np.clip(predictions, 1, 5)
    
    return dict(zip(unrated_movies, predictions))


def get_content_scores(user_id, train_df, movies_df,
                       cosine_sim, movie_indices, rated_movie_ids):
    """
    Get accumulated content based scores for all unrated movies.
    Based on genre similarity to user's highly rated movies.
    Returns dict: {movie_id: accumulated_similarity_score}
    """
    # Only use movies user rated 4 or higher as positive signals
    user_ratings = train_df[
        (train_df['user_id'] == user_id) &
        (train_df['rating'] >= 4)
    ].copy()
    
    if user_ratings.empty:
        return {}
    
    # Merge to get movie titles for similarity matrix lookup
    user_ratings = pd.merge(
        user_ratings[['user_id', 'movie_id', 'rating']],
        movies_df[['movie_id', 'title']],
        on='movie_id',
        how='inner'
    )
    
    # Initialize score array — one slot per movie
    scores = np.zeros(len(movies_df))
    
    # Accumulate similarity scores from each highly rated movie
    # Movies similar to many liked movies get higher accumulated scores
    for _, row in user_ratings.iterrows():
        title = row['title']
        if title in movie_indices:
            idx = movie_indices[title]
            scores += cosine_sim[idx]
    
    # Convert to dictionary — only include unrated movies
    result = {}
    for i, movie_id in enumerate(movies_df['movie_id'].values):
        if movie_id not in rated_movie_ids:
            result[movie_id] = scores[i]
    
    return result


def get_hybrid_scores(user_id, train_df, movies_df, svd_model, ncf_model,
                      user_encoder, movie_encoder, cosine_sim, movie_indices,
                      svd_weight, ncf_weight, content_weight):
    """
    Get normalized hybrid scores combining all three models.
    MinMaxScaler normalizes all scores to 0-1 before combining.
    Fixes scale mismatch: SVD/NCF (1-5) vs Content (0-382).
    Returns dict: {movie_id: hybrid_score}
    """
    # Get movies user has already rated — exclude from recommendations
    rated_movie_ids = set(
        train_df[train_df['user_id'] == user_id]['movie_id'].values
    )
    
    # Get raw scores from each model independently
    svd_scores = get_svd_scores(user_id, svd_model, movies_df, rated_movie_ids)
    ncf_scores = get_ncf_scores(user_id, ncf_model, movies_df, rated_movie_ids,
                                 user_encoder, movie_encoder)
    content_scores = get_content_scores(user_id, train_df, movies_df,
                                         cosine_sim, movie_indices, rated_movie_ids)
    
    # Intersection — only movies all three models can score
    # Ensures every recommendation has signal from all three models
    common_movies = (set(svd_scores.keys()) &
                     set(ncf_scores.keys()) &
                     set(content_scores.keys()))
    
    if not common_movies:
        return {}
    
    # Sort for consistent array ordering
    movie_list = sorted(list(common_movies))
    
    # Extract scores in same order for alignment
    svd_arr = np.array([svd_scores[m] for m in movie_list]).reshape(-1, 1)
    ncf_arr = np.array([ncf_scores[m] for m in movie_list]).reshape(-1, 1)
    content_arr = np.array([content_scores[m] for m in movie_list]).reshape(-1, 1)
    
    # Normalize all scores to 0-1 — fixes scale mismatch problem
    scaler = MinMaxScaler()
    svd_norm = scaler.fit_transform(svd_arr).flatten()
    ncf_norm = scaler.fit_transform(ncf_arr).flatten()
    content_norm = scaler.fit_transform(content_arr).flatten()
    
    # Weighted combination — weights from tuned hybrid config
    hybrid = (svd_weight * svd_norm +
              ncf_weight * ncf_norm +
              content_weight * content_norm)
    
    return dict(zip(movie_list, hybrid))


def get_recommendations(user_id, model_name, n, train_df, movies_df,
                        svd_model, ncf_model, user_encoder, movie_encoder,
                        cosine_sim, movie_indices, hybrid_config):
    """
    Master recommendation function.
    Routes to correct model based on model_name.
    Returns dataframe of top N recommendations with scores.
    """
    # Get movies user has already rated
    rated_movie_ids = set(
        train_df[train_df['user_id'] == user_id]['movie_id'].values
    )
    
    # Route to correct scoring function based on selected model
    if model_name == "SVD":
        scores = get_svd_scores(user_id, svd_model, movies_df, rated_movie_ids)
        score_label = "Predicted Rating"
        
    elif model_name == "Neural CF":
        scores = get_ncf_scores(user_id, ncf_model, movies_df, rated_movie_ids,
                                 user_encoder, movie_encoder)
        score_label = "Predicted Rating"
        
    elif model_name == "Content Based":
        scores = get_content_scores(user_id, train_df, movies_df,
                                     cosine_sim, movie_indices, rated_movie_ids)
        score_label = "Similarity Score"
        
    else:  # Hybrid
        scores = get_hybrid_scores(
            user_id, train_df, movies_df, svd_model, ncf_model,
            user_encoder, movie_encoder, cosine_sim, movie_indices,
            hybrid_config['svd_weight'],
            hybrid_config['ncf_weight'],
            hybrid_config['content_weight']
        )
        score_label = "Hybrid Score"
    
    if not scores:
        return None, score_label
    
    # Sort by score descending and take top N
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    top_ids = [m for m, _ in top_movies]
    top_scores = [round(float(s), 4) for _, s in top_movies]
    
    # Build results dataframe with movie titles and genres
    results = pd.DataFrame({
        'movie_id': top_ids,
        'score': top_scores
    })
    results = results.merge(
        movies_df[['movie_id', 'title', 'genres']],
        on='movie_id',
        how='left'
    )
    
    return results, score_label


def get_similar_movies(title, cosine_sim, movie_indices, movies_df, n=10):
    """
    Find movies similar to a given title.
    Uses content based cosine similarity.
    Returns dataframe of N most similar movies.
    """
    if title not in movie_indices:
        return None
    
    # Get row index of this movie in similarity matrix
    idx = movie_indices[title]
    
    # Get similarity scores with all other movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the input movie itself from results
    sim_scores = [
        (i, score) for i, score in sim_scores
        if movies_df.iloc[i]['title'] != title
    ][:n]
    
    movie_idx = [i[0] for i in sim_scores]
    similarity = [round(i[1], 4) for i in sim_scores]
    
    result = movies_df.iloc[movie_idx][['title', 'genres']].copy()
    result['similarity_score'] = similarity
    
    return result.reset_index(drop=True)


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================
def render_movie_card(rank, title, genres, score, score_label):
    """
    Renders a styled movie card using HTML.
    Each genre gets its own colored badge.
    """
    # Create individual genre badges
    genre_list = genres.split('|')
    genre_badges = ''.join([
        f'<span class="genre-badge">{g}</span>'
        for g in genre_list
    ])
    
    # Build full card HTML
    card_html = f"""
    <div class="movie-card">
        <span class="rank-number">#{rank}</span>
        <div class="movie-title">{title}</div>
        <div style="margin: 8px 0">{genre_badges}</div>
        <div class="score-display">
            {score_label}: {score}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_metrics_table(eval_results):
    """
    Renders model performance comparison table.
    Highlights best values in blue.
    """
    st.markdown(
        '<div class="section-header">📊 Model Performance Comparison</div>',
        unsafe_allow_html=True
    )
    
    if eval_results is not None:
        # Highlight maximum values in metric columns
        metric_cols = [c for c in eval_results.columns
                       if c not in ['model', 'rmse', 'mae']]
        styled_df = eval_results.style.highlight_max(
            subset=metric_cols,
            color='#1a3a6b'
        )
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Evaluation results not found. Run 06_evaluation.ipynb first.")


# ============================================================
# LOAD MODELS AND DATA
# Runs once when app starts — cached after first load
# ============================================================
with st.spinner("🔄 Loading models... please wait"):
    (svd_model, ncf_model, user_encoder, movie_encoder,
     cosine_sim, movie_indices, hybrid_config) = load_all_models()
    train, movies, eval_results = load_data()

# ============================================================
# SIDEBAR UI
# All user inputs go in the sidebar
# ============================================================
with st.sidebar:
    st.markdown("# 🎬 Movie Recommender")
    st.markdown("*SVD · Neural CF · Content Based · Hybrid*")
    st.markdown("---")
    
    # Mode selection
    st.markdown("### 🎯 Mode")
    mode = st.radio(
        label="Mode",
        options=[
            "🎯 Get Recommendations",
            "🔍 Find Similar Movies"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if mode == "🎯 Get Recommendations":
        
        # Model selection
        st.markdown("### 🤖 Model")
        model_name = st.selectbox(
            label="Model",
            options=["Hybrid (Tuned)", "SVD", "Neural CF", "Content Based"],
            label_visibility="collapsed"
        )
        
        # Map display name to internal model name
        model_map = {
            "Hybrid (Tuned)": "Hybrid",
            "SVD": "SVD",
            "Neural CF": "Neural CF",
            "Content Based": "Content Based"
        }
        
        st.markdown("---")
        
        # User ID input
        st.markdown("### 👤 User ID")
        st.caption("Valid range: 1 — 6040")
        user_id = st.number_input(
            label="User ID",
            min_value=1,
            max_value=6040,
            value=1680,      # default to our consistent test user
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Number of recommendations
        st.markdown("### 🔢 Number of Recommendations")
        n_recs = st.slider(
            label="N",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Generate button
        generate_button = st.button(
            "🎬 Get Recommendations",
            use_container_width=True,
            type="primary"
        )
        
    else:  # Find Similar Movies mode
        
        # Movie title dropdown
        st.markdown("### 🎬 Movie Title")
        all_titles = sorted(movies['title'].tolist())
        selected_title = st.selectbox(
            label="Movie Title",
            options=all_titles,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Number of similar movies
        st.markdown("### 🔢 Similar Movies Count")
        n_similar = st.slider(
            label="N Similar",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Find button
        find_button = st.button(
            "🔍 Find Similar Movies",
            use_container_width=True,
            type="primary"
        )
    
    # Hybrid weights info at bottom of sidebar
    st.markdown("---")
    st.markdown("### ⚙️ Hybrid Weights")
    st.markdown(f"SVD: **{hybrid_config['svd_weight']}**")
    st.markdown(f"NCF: **{hybrid_config['ncf_weight']}**")
    st.markdown(f"Content: **{hybrid_config['content_weight']}**")


# ============================================================
# MAIN CONTENT AREA
# ============================================================

# App header
st.markdown(
    '<h1 style="color:#4f8bf9; text-align:center;">'
    '🎬 Movie Recommendation System</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center; color:#888; font-size:16px;">'
    'Powered by SVD · Neural CF · Content Based Filtering · Hybrid Model'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================
# GET RECOMMENDATIONS MODE
# ============================================================
if mode == "🎯 Get Recommendations":
    
    # Split page into two columns
    # col1 = watch history (1/3 width)
    # col2 = recommendations (2/3 width)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(
            '<div class="section-header">📚 Watch History</div>',
            unsafe_allow_html=True
        )
        
        # Get user's top 5 highly rated movies from train set
        user_filtered = train[
            (train['user_id'] == user_id) &
            (train['rating'] >= 4)
        ][['user_id', 'movie_id', 'rating']].copy()

        user_history = pd.merge(
            user_filtered,
            movies[['movie_id', 'title', 'genres']],
            on='movie_id',
            how='inner'
        ).sort_values('rating', ascending=False).head(5)
        
        if user_history.empty:
            st.warning(
                f"User {user_id} has no highly rated movies "
                f"in training set. Try a different user ID."
            )
        else:
            # Render small history cards with orange accent
            for _, row in user_history.iterrows():
                st.markdown(f"""
                <div style="background:#1e2130; border-radius:8px;
                            padding:10px; margin:5px 0;
                            border-left:3px solid #f9a84f;">
                    <div style="color:#fff; font-size:13px;
                                font-weight:600;">{row['title']}</div>
                    <div style="color:#f9a84f; font-size:12px;">
                        ⭐ {row['rating']}/5
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            f'<div class="section-header">'
            f'🎯 {model_map[model_name]} Recommendations'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if generate_button:
            # Show spinner while generating recommendations
            with st.spinner(
                f"Generating {n_recs} recommendations "
                f"using {model_name}..."
            ):
                results, score_label = get_recommendations(
                    user_id=user_id,
                    model_name=model_map[model_name],
                    n=n_recs,
                    train_df=train,
                    movies_df=movies,
                    svd_model=svd_model,
                    ncf_model=ncf_model,
                    user_encoder=user_encoder,
                    movie_encoder=movie_encoder,
                    cosine_sim=cosine_sim,
                    movie_indices=movie_indices,
                    hybrid_config=hybrid_config
                )
            
            if results is None:
                st.error(
                    f"Could not generate recommendations for "
                    f"User {user_id}. Try a different user ID or model."
                )
            else:
                st.success(
                    f"✅ Top {len(results)} recommendations "
                    f"for User {user_id} using {model_name}!"
                )
                
                # Render each recommendation as a styled card
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    render_movie_card(
                        rank=rank,
                        title=row['title'],
                        genres=row['genres'],
                        score=row['score'],
                        score_label=score_label
                    )
        else:
            # Placeholder before button is clicked
            st.info(
                "👈 Select a model and User ID, "
                "then click **Get Recommendations**!"
            )

# ============================================================
# FIND SIMILAR MOVIES MODE
# ============================================================
else:
    
    st.markdown(
        '<div class="section-header">🔍 Find Similar Movies</div>',
        unsafe_allow_html=True
    )
    
    if find_button:
        
        with st.spinner(
            f"Finding movies similar to '{selected_title}'..."
        ):
            similar = get_similar_movies(
                title=selected_title,
                cosine_sim=cosine_sim,
                movie_indices=movie_indices,
                movies_df=movies,
                n=n_similar
            )
        
        if similar is None:
            st.error(f"Movie '{selected_title}' not found.")
        else:
            # Show selected movie info card with orange accent
            selected_movie = movies[
                movies['title'] == selected_title
            ].iloc[0]
            
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2130,#252840);
                        border-radius:12px; padding:20px; margin-bottom:20px;
                        border-left:4px solid #f9a84f;">
                <div style="color:#f9a84f; font-size:12px;
                            font-weight:600;">SELECTED MOVIE</div>
                <div style="color:#fff; font-size:20px;
                            font-weight:700; margin:5px 0;">
                    {selected_title}
                </div>
                <div style="color:#888; font-size:13px;">
                    {selected_movie['genres']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"✅ Found {len(similar)} similar movies!")
            
            # Render similar movies as cards
            for rank, (_, row) in enumerate(similar.iterrows(), 1):
                render_movie_card(
                    rank=rank,
                    title=row['title'],
                    genres=row['genres'],
                    score=row['similarity_score'],
                    score_label="Similarity Score"
                )
    else:
        st.info(
            "👈 Select a movie title and click "
            "**Find Similar Movies**!"
        )

# ============================================================
# MODEL PERFORMANCE TABLE
# Always visible at bottom of page
# ============================================================
st.markdown("---")
render_metrics_table(eval_results)

# ============================================================
# ABOUT SECTION
# ============================================================
st.markdown("---")
st.markdown(
    '<div class="section-header">📖 About the Models</div>',
    unsafe_allow_html=True
)

# Four columns for four model descriptions
a1, a2, a3, a4 = st.columns(4)

with a1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#4f8bf9">SVD</h4>
        <p style="color:#888; font-size:12px;">
        Matrix factorization compressing the user-movie matrix 
        into latent factors. Fast and reliable on explicit ratings.
        </p>
        <p style="color:#4f8bf9; font-size:12px;">
        RMSE: 0.9372 | P@10: 0.0837
        </p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#4f8bf9">Neural CF</h4>
        <p style="color:#888; font-size:12px;">
        Deep learning model replacing SVD dot product with 
        neural layers to capture non-linear interactions.
        </p>
        <p style="color:#4f8bf9; font-size:12px;">
        RMSE: 0.9587 | P@10: 0.0980
        </p>
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#4f8bf9">Content Based</h4>
        <p style="color:#888; font-size:12px;">
        Recommends movies similar to liked ones using 
        TF-IDF genre vectors and cosine similarity.
        </p>
        <p style="color:#4f8bf9; font-size:12px;">
        P@10: 0.0153 | NDCG@10: 0.0186
        </p>
    </div>
    """, unsafe_allow_html=True)

with a4:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#4f8bf9">Hybrid (Tuned)</h4>
        <p style="color:#888; font-size:12px;">
        Combines all three with optimized weights 
        SVD=0.35, NCF=0.50, Content=0.15.
        Best overall recommendation quality.
        </p>
        <p style="color:#4f8bf9; font-size:12px;">
        P@10: 0.1041 | NDCG@10: 0.1018
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#444; font-size:12px;">'
    'Movie Recommendation System | MovieLens 1M Dataset | '
    'SVD + Neural CF + Content Based + Hybrid'
    '</p>',
    unsafe_allow_html=True
)
