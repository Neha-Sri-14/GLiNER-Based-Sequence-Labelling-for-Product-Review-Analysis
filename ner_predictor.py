from pathlib import Path
import re

from gliner import GLiNER


class NERPredictor:

    def __init__(self, model_path=None, threshold=0.5):

        # Default model path:
        # ProductReview_GLiner/gliner_model/model
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent
                / "gliner_model"
                / "model"
            )

        model_path = Path(model_path)

        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned GLiNER model directory not found: {model_path}"
            )

        print(f"Loading fine-tuned GLiNER model from: {model_path}")

        self.model = GLiNER.from_pretrained(
            str(model_path),
            load_tokenizer=True
        )

        # Exact labels used during training
        self.labels = [
            "brand",
            "product",
            "feature",
            "issue",
            "opinion"
        ]

        self.threshold = threshold

        print("GLiNER model loaded successfully.")
        print(f"Labels: {self.labels}")
        print(f"Prediction threshold: {self.threshold}")

    def predict(self, text):

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        text = text.strip()

        if not text:
            return []

        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold
        )

        return sorted(
            entities,
            key=lambda x: x.get("start", 0)
        )

    def get_entities(self, text):

        predictions = self.predict(text)

        entities = []

        for entity in predictions:

            entities.append({
                "text": entity.get("text", ""),
                "label": entity.get("label", "").upper(),
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "score": round(
                    float(entity.get("score", 0)),
                    4
                )
            })

        return entities

    def predict_bio(self, text):

        entities = self.get_entities(text)

        tokens = []
        token_offsets = []

        for match in re.finditer(r"\S+", text):
            tokens.append(match.group())
            token_offsets.append(
                (match.start(), match.end())
            )

        bio_sequence = []

        for token, (token_start, token_end) in zip(
            tokens,
            token_offsets
        ):

            label = "O"

            for entity in entities:

                entity_start = entity["start"]
                entity_end = entity["end"]

                if (
                    token_start < entity_end
                    and token_end > entity_start
                ):

                    if token_start <= entity_start:
                        label = f"B-{entity['label']}"
                    else:
                        label = f"I-{entity['label']}"

                    break

            bio_sequence.append(
                (token, label)
            )

        return {
            "tokens": tokens,
            "bio_sequence": bio_sequence,
            "entities": entities
        }