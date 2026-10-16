from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ner_predictor import NERPredictor
from aspect_sentiment import AspectSentimentAnalyzer
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load the fine-tuned GLiNER model once when Flask starts.
MODEL_PATH = BASE_DIR / "gliner_model" / "model"

ner_predictor = NERPredictor(model_path=MODEL_PATH)
sentiment_analyzer = AspectSentimentAnalyzer()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_review():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("review", "").strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "Please enter a product review."
            }), 400

        # 1. GLiNER span/entity prediction
        entities = ner_predictor.get_entities(text)

        # 2. BIO sequence labelling
        bio_result = ner_predictor.predict_bio(text)

        # 3. Aspect-level + overall sentiment
        analysis = sentiment_analyzer.analyze(
            text=text,
            entities=entities
        )

        return jsonify({
            "success": True,
            "review": text,
            "entities": entities,
            "tokens": bio_result["tokens"],
            "bio_sequence": [
                {
                    "token": token,
                    "label": label
                }
                for token, label in bio_result["bio_sequence"]
            ],
            "aspect_sentiments": analysis.get(
                "aspect_sentiments", []
            ),
            "overall_sentiment": analysis.get(
                "overall_sentiment", {}
            ),
            "explanation": analysis.get(
                "explanation", ""
            )
        })

    except Exception as exc:
        app.logger.exception("Review analysis failed")
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "labels": ner_predictor.labels
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
