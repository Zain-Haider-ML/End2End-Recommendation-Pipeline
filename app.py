from flask import (Flask, jsonify, redirect, render_template_string, request,
                   url_for)

from models import get_movie_title, predict_rating, recommend

import os

app = Flask(__name__)


# Home UI
@app.route("/", methods=["GET", "POST"])
def home():
    html = """
    <h1>🎬 Movie Recommendation User Interface Chomoloko</h1>
    
    <form action="/do_predict" method="post">
        <h2>🔹 Predict Single Movie Rating</h2>
        <label>User ID:</label><input type="number" name="user_id" required><br><br>
        <label>Movie ID:</label><input type="number" name="movie_id" required><br><br>
        <label>Model:</label>
        <select name="model">
            <option value="meta">Meta</option>
            <option value="weighted">Weighted</option>
            <option value="switching">Switching</option>
            <option value="svd">SVD</option>
            <option value="feature">Feature Combination</option>
        </select><br><br>
        <button type="submit">Predict Rating</button>
    </form>

    <hr>

    <form action="/do_recommend" method="post">
        <h2>🔹 Recommend Movies</h2>
        <label>User ID:</label><input type="number" name="user_id" required><br><br>
        <label>Model:</label>
        <select name="model">
            <option value="meta">Meta</option>
            <option value="weighted">Weighted</option>
            <option value="switching">Switching</option>
            <option value="mixed">Mixed</option>
            <option value="cascade">Cascade</option>
        </select><br><br>
        <label>Top N:</label><input type="number" name="top_n" value="5" required><br><br>
        <button type="submit">Recommend</button>
    </form>
    """
    return render_template_string(html)


# API: Predict Rating (actual endpoint)
@app.route("/predict", methods=["GET"])
def predict():
    user_id = int(request.args.get("user_id"))
    movie_id = int(request.args.get("movie_id"))
    model = request.args.get("model", "meta")
    alpha = float(request.args.get("alpha", 0.5))

    try:
        score = predict_rating(user_id, movie_id, model=model, alpha=alpha)
        return jsonify(
            {
                "user_id": user_id,
                "movie_id": movie_id,
                "model": model,
                "predicted_rating": score,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# API: Recommend Movies (actual endpoint)
@app.route("/recommend", methods=["GET"])
def recommend_api():
    user_id = int(request.args.get("user_id"))
    model = request.args.get("model", "meta")
    top_n = int(request.args.get("top_n", 5))
    alpha = float(request.args.get("alpha", 0.5))

    try:
        recs = recommend(user_id, model=model, top_n=top_n, alpha=alpha, verbose=False)

        if isinstance(recs, dict):  # For mixed models
            output = {
                "collaborative": [
                    get_movie_title(mid) for mid in recs["collaborative"]
                ],
                "content": [get_movie_title(mid) for mid in recs["content"]],
            }
        else:
            output = [
                {"movie_id": mid, "title": get_movie_title(mid), "score": score}
                for mid, score in recs
            ]

        return jsonify({"user_id": user_id, "model": model, "recommendations": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Form Action: Do Predict (handles form submission)
@app.route("/do_predict", methods=["POST"])
def do_predict():
    user_id = request.form.get("user_id")
    movie_id = request.form.get("movie_id")
    model = request.form.get("model", "meta")

    return redirect(url_for("predict", user_id=user_id, movie_id=movie_id, model=model))


# Form Action: Do Recommend (handles form submission)
@app.route("/do_recommend", methods=["POST"])
def do_recommend():
    user_id = request.form.get("user_id")
    model = request.form.get("model", "meta")
    top_n = request.form.get("top_n", 5)

    return redirect(url_for("recommend_api", user_id=user_id, model=model, top_n=top_n))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(debug=True, host="0.0.0.0", port=port)
