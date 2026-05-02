from flask import Flask, render_template, request, jsonify
from recommender import recommend, recommend_by_selected_genre

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def rec():
    return jsonify(recommend(request.json.get("title")))

@app.route("/genre", methods=["POST"])
def genre():
    return jsonify(recommend_by_selected_genre(request.json.get("genre")))

if __name__ == "__main__":
    app.run(debug=True)