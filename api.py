# api.py
from flask import Flask, request, jsonify
from models import recommend, predict_rating, get_movie_title

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    user_id = int(request.args.get("user_id"))
    movie_id = int(request.args.get("movie_id"))
    model = request.args.get("model", "meta")
    alpha = float(request.args.get("alpha", 0.5))
    try:
        score = predict_rating(user_id, movie_id, model=model, alpha=alpha)
        return jsonify({"user_id": user_id, "movie_id": movie_id, "model": model, "score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/recommend", methods=["GET"])
def recommend_api():
    user_id = int(request.args.get("user_id"))
    model = request.args.get("model", "meta")
    top_n = int(request.args.get("top_n", 5))
    alpha = float(request.args.get("alpha", 0.5))
    try:
        recs = recommend(user_id, model=model, top_n=top_n, alpha=alpha, verbose=False)
        if isinstance(recs, dict):  # mixed model
            output = {
                "collaborative": [get_movie_title(mid) for mid in recs['collaborative']],
                "content": [get_movie_title(mid) for mid in recs['content']]
            }
        else:
            output = [
                {"movie_id": mid, "title": get_movie_title(mid), "score": score}
                for mid, score in recs
            ]
        return jsonify({"user_id": user_id, "model": model, "recommendations": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
