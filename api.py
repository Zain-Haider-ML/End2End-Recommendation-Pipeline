# app.py
import streamlit as st

from models import get_movie_title, recommend

st.set_page_config(page_title="Hybrid Recommender", layout="centered")
st.title("🎬 Movie Recommender System")

user_id = st.number_input("Enter User ID", min_value=1, max_value=943, value=1)
model = st.selectbox(
    "Select Recommender Model",
    ["meta", "weighted", "switching", "feature", "svd", "cbf", "mixed", "cascade"],
)
top_n = st.slider("Top N Recommendations", 1, 20, 5)
alpha = (
    st.slider("Alpha (Weighting Factor for Weighted Hybrid)", 0.0, 1.0, 0.5)
    if model == "weighted"
    else None
)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        try:
            recs = recommend(
                user_id, model=model, top_n=top_n, alpha=alpha or 0.5, verbose=False
            )
            if isinstance(recs, dict):  # mixed model returns dict
                st.subheader("Collaborative Filtering Recommendations")
                for movie_id in recs["collaborative"]:
                    title = get_movie_title(movie_id)
                    st.write(f"\- {title}")
                st.subheader("Content-Based Recommendations")
                for movie_id in recs["content"]:
                    title = get_movie_title(movie_id)
                    st.write(f"\- {title}")
            else:
                st.subheader("Top Recommendations")
                for movie_id, score in recs:
                    title = get_movie_title(movie_id)
                    st.write(f"\- **{title}** → Predicted Score: {score:.2f}")
        except Exception as e:
            st.error(f"Failed to get recommendations: {e}")
