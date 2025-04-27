import pandas as pd
import numpy as np
import os
import zipfile
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Download
url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
file_path = "ml-100k.zip"
if not os.path.exists("ml-100k"):
    urllib.request.urlretrieve(url, file_path)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(".")

# Load ratings
ratings = pd.read_csv(
    "ml-100k/u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"]
)

# Load movie titles and genres
movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    names=[
        "movieId",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ],
)

# Combine genres into a string column for content-based
genre_cols = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

movies["genres"] = movies[genre_cols].apply(
    lambda row: " ".join([col for col in genre_cols if row[col] == 1]), axis=1
)

# Drop unnecessary columns
movies = movies[["movieId", "title", "genres"]]


# Surprise setup
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

# TF-IDF for content-based
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies["movieId"])


def weighted_hybrid(user_id, movie_id, alpha=0.5):
    # CF score from SVD
    cf_score = svd.predict(user_id, movie_id).est

    # CBF score from cosine similarity
    idx = movie_indices.get(movie_id, None)
    if idx is None:
        cbf_score = 0
    else:
        cbf_score = np.mean(cos_sim[idx])

    # Weighted average
    return alpha * cf_score + (1 - alpha) * cbf_score


def switching_hybrid(user_id, movie_id):
    # Rule: new users (cold start) use CBF
    user_ratings = ratings[ratings["userId"] == user_id]
    if len(user_ratings) < 5:
        # Use content-based
        idx = movie_indices.get(movie_id, None)
        return np.mean(cos_sim[idx]) if idx is not None else 0
    else:
        # Use collaborative filtering
        return svd.predict(user_id, movie_id).est


def mixed_hybrid(user_id, top_n=5):
    # Top N recommendations from CF
    movie_scores = []
    for movie_id in movies["movieId"]:
        est = svd.predict(user_id, movie_id).est
        movie_scores.append((movie_id, est))
    cf_recs = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # Top N recommendations from CBF
    watched_movies = ratings[ratings["userId"] == user_id]["movieId"]
    recs = set()
    for mid in watched_movies:
        idx = movie_indices.get(mid, None)
        if idx is not None:
            sim_scores = list(enumerate(cos_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            for i, score in sim_scores[1 : top_n + 1]:
                rec_id = movies.iloc[i]["movieId"]
                recs.add((rec_id, score))
    cbf_recs = sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]

    # Return both
    return {
        "collaborative": [x[0] for x in cf_recs],
        "content": [x[0] for x in cbf_recs],
    }


def cascade_hybrid(user_id, top_n=10):
    # Step 1: get CF top-N
    candidates = []
    for movie_id in movies["movieId"]:
        est = svd.predict(user_id, movie_id).est
        candidates.append((movie_id, est))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]

    # Step 2: rerank with content similarity
    reranked = []
    for movie_id, score in candidates:
        idx = movie_indices.get(movie_id, None)
        cbf_score = np.mean(cos_sim[idx]) if idx is not None else 0
        reranked.append((movie_id, cbf_score))

    # Final recommendation
    return sorted(reranked, key=lambda x: x[1], reverse=True)


# Train a hybrid model using both CF and CBF scores as features
def build_feature_combination_model():
    X, y = [], []
    for _, row in ratings.iterrows():
        user_id, movie_id, rating = row["userId"], row["movieId"], row["rating"]
        cf_score = svd.predict(user_id, movie_id).est
        idx = movie_indices.get(movie_id, None)
        cbf_score = np.mean(cos_sim[idx]) if idx is not None else 0
        X.append([cf_score, cbf_score])
        y.append(rating)

    X, y = np.array(X), np.array(y)
    reg = LinearRegression().fit(X, y)
    return reg


hybrid_reg = build_feature_combination_model()


def feature_combination_predict(user_id, movie_id):
    cf = svd.predict(user_id, movie_id).est
    idx = movie_indices.get(movie_id, None)
    cbf = np.mean(cos_sim[idx]) if idx is not None else 0
    return hybrid_reg.predict(np.array([[cf, cbf]]))[0]


# Step 1: Use CF model predictions as features for another model
def build_meta_model():
    X, y = [], []
    for _, row in ratings.iterrows():
        user_id, movie_id, rating = row["userId"], row["movieId"], row["rating"]
        cf_score = svd.predict(user_id, movie_id).est

        # Use content features (TF-IDF vector mean for movie)
        idx = movie_indices.get(movie_id, None)
        content_feat = (
            tfidf_matrix[idx].toarray()[0]
            if idx is not None
            else np.zeros(tfidf_matrix.shape[1])
        )

        features = np.concatenate([[cf_score], content_feat])
        X.append(features)
        y.append(rating)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    return rf


meta_model = build_meta_model()


def meta_level_predict(user_id, movie_id):
    cf_score = svd.predict(user_id, movie_id).est
    idx = movie_indices.get(movie_id, None)
    content_feat = (
        tfidf_matrix[idx].toarray()[0]
        if idx is not None
        else np.zeros(tfidf_matrix.shape[1])
    )
    features = np.concatenate([[cf_score], content_feat])
    return meta_model.predict([features])[0]


# === Wrapper: Single-prediction dispatcher ===
def predict_rating(user_id, movie_id, model="meta", alpha=0.5):
    if model == "meta":
        return meta_level_predict(user_id, movie_id)
    elif model == "weighted":
        return weighted_hybrid(user_id, movie_id, alpha=alpha)
    elif model == "switching":
        return switching_hybrid(user_id, movie_id)
    elif model == "feature":
        return feature_combination_predict(user_id, movie_id)
    elif model == "svd":
        return svd.predict(user_id, movie_id).est
    elif model == "cbf":
        idx = movie_indices.get(movie_id, None)
        return np.mean(cos_sim[idx]) if idx is not None else 0
    else:
        raise ValueError(f"Unknown model: {model}")


# === Top-N Recommender ===
def recommend(user_id, model="meta", top_n=10, alpha=0.5, verbose=True):
    if model == "mixed":
        return mixed_hybrid(user_id, top_n=top_n)
    elif model == "cascade":
        return cascade_hybrid(user_id, top_n=top_n)

    rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    movie_ids = movies["movieId"].unique()
    preds = []

    for movie_id in movie_ids:
        if movie_id in rated_movies:
            continue
        try:
            score = predict_rating(user_id, movie_id, model=model, alpha=alpha)
            preds.append((movie_id, score))
        except Exception:
            continue

    top = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

    if verbose:
        print(f"Top {top_n} recommendations for User {user_id} using '{model}' model:")
        for movie_id, score in top:
            title = movies[movies["movieId"] == movie_id]["title"].values[0]
            print(f"  • {title} (MovieID {movie_id}) → Predicted Rating: {score:.2f}")

    return top


def get_movie_title(movie_id):
    return movies[movies["movieId"] == movie_id]["title"].values[0]


# Get top 5 using Meta-Level Hybrid
recommend(1, model="meta", top_n=5)

# Get top 5 using Weighted Hybrid
recommend(1, model="weighted", top_n=5, alpha=0.7)

# Get top 5 using pure Collaborative Filtering (SVD)
recommend(1, model="switching", top_n=5)

recommend(1, model="hybrid", top_n=5)

recommend(1, model="svd", top_n=5)

recommend(1, model="feature", top_n=5)
