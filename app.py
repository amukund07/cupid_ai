from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

features = [
    "initiates_conversation",
    "engagement",
    "quick_responses",
    "eye_contact",
    "joke_responses",
    "nervous",
    "stays_near_you",
    "helps_you",
    "smiles",
    "takes_you_out"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    score = None
    if request.method == "POST":
        data = {f: int(request.form[f]) for f in features}
        df = pd.DataFrame([data])
        score = round(model.predict(df)[0], 2)

    return render_template("predict.html", features=features, score=score)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use Railway PORT if available, else 5000
    app.run(host="0.0.0.0", port=port, debug=True)
