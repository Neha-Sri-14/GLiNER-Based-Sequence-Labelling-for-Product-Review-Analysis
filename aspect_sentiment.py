import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------
# Download VADER lexicon if it is not already available
# ---------------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class AspectSentimentAnalyzer:

    def __init__(self):

        self.vader = SentimentIntensityAnalyzer()

        # Product-review aspects used as a fallback
        # when GLiNER does not detect a feature.
        self.fallback_aspects = [
            "battery",
            "battery life",
            "camera",
            "camera quality",
            "screen",
            "display",
            "sound",
            "sound quality",
            "audio",
            "speaker",
            "speakers",
            "microphone",
            "performance",
            "speed",
            "design",
            "build quality",
            "quality",
            "price",
            "cost",
            "value",
            "charging",
            "charging speed",
            "charger",
            "storage",
            "memory",
            "keyboard",
            "trackpad",
            "touchpad",
            "comfort",
            "ear cushions",
            "noise cancellation",
            "connectivity",
            "wireless connectivity",
            "bluetooth",
            "signal",
            "reception",
            "network",
            "call quality",
            "software",
            "customer service",
            "accessories",
            "case",
            "headset",
            "ringtones",
            "menu",
            "voice dialing",
            "speed dialing",
            "games"
        ]

    # ---------------------------------------------------------
    # Normalize GLiNER entities
    # ---------------------------------------------------------
    def normalize_entities(self, entities):

        normalized = []

        if not entities:
            return normalized

        for entity in entities:

            label = str(
                entity.get("label", "")
            ).upper().strip()

            text = str(
                entity.get("text", "")
            ).strip()

            if not text:
                continue

            if label not in {
                "BRAND",
                "PRODUCT",
                "FEATURE",
                "ISSUE",
                "OPINION"
            }:
                continue

            normalized.append({
                "text": text,
                "label": label,
                "start": int(entity.get("start", 0)),
                "end": int(entity.get("end", 0)),
                "score": float(
                    entity.get("score", 0)
                )
            })

        return sorted(
            normalized,
            key=lambda x: x["start"]
        )

    # ---------------------------------------------------------
    # Split review into sentences
    # ---------------------------------------------------------
    def split_sentences(self, text):

        sentences = []

        for match in re.finditer(
            r"[^.!?]+(?:[.!?]+|$)",
            text
        ):

            sentence_text = match.group().strip()

            if not sentence_text:
                continue

            # Locate actual non-space start/end
            start = match.start()

            while (
                start < match.end()
                and text[start].isspace()
            ):
                start += 1

            end = start + len(sentence_text)

            sentences.append({
                "text": sentence_text,
                "start": start,
                "end": end
            })

        return sentences

    # ---------------------------------------------------------
    # Find sentence containing an entity
    # ---------------------------------------------------------
    def get_sentence_for_entity(
        self,
        entity,
        sentences
    ):

        entity_start = entity["start"]
        entity_end = entity["end"]

        for sentence in sentences:

            if (
                entity_start < sentence["end"]
                and entity_end > sentence["start"]
            ):
                return sentence

        return None

    # ---------------------------------------------------------
    # Fallback feature extraction
    # ---------------------------------------------------------
    def extract_fallback_aspects(
        self,
        text,
        existing_entities
    ):

        aspects = []

        existing_ranges = [
            (
                entity["start"],
                entity["end"]
            )
            for entity in existing_entities
            if entity["label"] == "FEATURE"
        ]

        lower_text = text.lower()

        # Longest phrases first
        sorted_aspects = sorted(
            self.fallback_aspects,
            key=len,
            reverse=True
        )

        for aspect in sorted_aspects:

            pattern = r"\b" + re.escape(aspect) + r"\b"

            for match in re.finditer(
                pattern,
                lower_text
            ):

                start = match.start()
                end = match.end()

                # Do not duplicate GLiNER entities
                overlaps = any(
                    start < existing_end
                    and end > existing_start
                    for existing_start, existing_end
                    in existing_ranges
                )

                if overlaps:
                    continue

                aspects.append({
                    "text": text[start:end],
                    "label": "FEATURE",
                    "start": start,
                    "end": end,
                    "score": 0.0,
                    "source": "fallback"
                })

                existing_ranges.append(
                    (start, end)
                )

        return aspects

    # ---------------------------------------------------------
    # Calculate VADER sentiment
    # ---------------------------------------------------------
    def vader_sentiment(self, text):

        scores = self.vader.polarity_scores(text)

        compound = scores["compound"]

        if compound >= 0.05:
            label = "POSITIVE"

        elif compound <= -0.05:
            label = "NEGATIVE"

        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"]
        }

    # ---------------------------------------------------------
    # Pair FEATURE with OPINION / ISSUE
    # ---------------------------------------------------------
    def pair_aspects_and_opinions(
        self,
        text,
        entities
    ):

        features = [
            entity
            for entity in entities
            if entity["label"] == "FEATURE"
        ]

        opinions = [
            entity
            for entity in entities
            if entity["label"] in {
                "OPINION",
                "ISSUE"
            }
        ]

        sentences = self.split_sentences(text)

        pairs = []

        for opinion in opinions:

            sentence = self.get_sentence_for_entity(
                opinion,
                sentences
            )

            if sentence is None:
                continue

            same_sentence_features = [
                feature
                for feature in features
                if (
                    feature["start"] < sentence["end"]
                    and feature["end"] > sentence["start"]
                )
            ]

            if not same_sentence_features:
                continue

            # Find the closest feature to the opinion
            closest_feature = min(
                same_sentence_features,
                key=lambda feature:
                    abs(
                        feature["start"]
                        - opinion["start"]
                    )
            )

            distance = abs(
                closest_feature["start"]
                - opinion["start"]
            )

            pairs.append({
                "aspect": closest_feature,
                "opinion": opinion,
                "distance": distance
            })

        return pairs

    # ---------------------------------------------------------
    # Calculate sentiment for one aspect
    # ---------------------------------------------------------
    def calculate_aspect_sentiment(
        self,
        aspect,
        opinion
    ):

        # Explicit issue = negative evidence
        if opinion["label"] == "ISSUE":

            return {
                "label": "NEGATIVE",
                "score": -0.75,
                "reason": (
                    f"The review mentions an issue with "
                    f"{aspect['text']}."
                )
            }

        opinion_text = opinion["text"]

        sentiment = self.vader_sentiment(
            opinion_text
        )

        compound = sentiment["compound"]

        if compound >= 0.05:
            label = "POSITIVE"

        elif compound <= -0.05:
            label = "NEGATIVE"

        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "score": compound,
            "reason": (
                f"The opinion '{opinion_text}' "
                f"expresses {label.lower()} sentiment "
                f"toward {aspect['text']}."
            )
        }

    # ---------------------------------------------------------
    # Overall review sentiment
    # ---------------------------------------------------------
    def get_review_sentiment(self, text):

        return self.vader_sentiment(text)

    # ---------------------------------------------------------
    # Convert numerical product rating to sentiment
    # ---------------------------------------------------------
    def get_rating_sentiment(self, rating):

        if rating is None:
            return None

        try:
            rating = float(rating)

        except (
            ValueError,
            TypeError
        ):
            return None

        if rating <= 2:
            return "NEGATIVE"

        elif rating == 3:
            return "NEUTRAL"

        else:
            return "POSITIVE"

    # ---------------------------------------------------------
    # Calculate overall aspect sentiment
    # ---------------------------------------------------------
    def get_overall_sentiment(
        self,
        aspect_sentiments,
        review_sentiment
    ):

        positive_count = sum(
            1
            for item in aspect_sentiments
            if item["sentiment"] == "POSITIVE"
        )

        negative_count = sum(
            1
            for item in aspect_sentiments
            if item["sentiment"] == "NEGATIVE"
        )

        neutral_count = sum(
            1
            for item in aspect_sentiments
            if item["sentiment"] == "NEUTRAL"
        )

        total_aspects = (
            positive_count
            + negative_count
            + neutral_count
        )

        # If aspect-level analysis exists,
        # use it as the primary evidence.
        if total_aspects > 0:

            if (
                positive_count > 0
                and negative_count > 0
            ):
                label = "MIXED"

                explanation = (
                    "The review contains both "
                    "positive and negative opinions "
                    "about different product aspects."
                )

            elif positive_count > negative_count:
                label = "POSITIVE"

                explanation = (
                    "The review contains more "
                    "positive aspect-level opinions."
                )

            elif negative_count > positive_count:
                label = "NEGATIVE"

                explanation = (
                    "The review contains more "
                    "negative aspect-level opinions."
                )

            else:
                label = "NEUTRAL"

                explanation = (
                    "The review does not contain "
                    "a clearly dominant sentiment."
                )

            # Score based on aspect sentiment
            scores = [
                item["score"]
                for item in aspect_sentiments
            ]

            score = (
                sum(scores) / len(scores)
                if scores
                else 0.0
            )

        else:

            # Fall back to full-review VADER
            label = review_sentiment["label"]
            score = review_sentiment["compound"]

            explanation = (
                "Overall sentiment was determined "
                "from the complete review because "
                "no aspect-level opinion pair was found."
            )

        return {
            "label": label,
            "score": round(score, 4),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "explanation": explanation
        }

    # ---------------------------------------------------------
    # MAIN ANALYSIS FUNCTION
    # ---------------------------------------------------------
    def analyze(
        self,
        text,
        entities=None
    ):

        if not isinstance(text, str):
            raise TypeError(
                "Review text must be a string."
            )

        text = text.strip()

        if not text:
            return {
                "entities": [],
                "aspect_sentiments": [],
                "overall_sentiment": {
                    "label": "NEUTRAL",
                    "score": 0.0,
                    "explanation": "Empty review."
                },
                "explanation": "No review text was provided."
            }

        # -----------------------------------------------------
        # 1. Normalize GLiNER entities
        # -----------------------------------------------------
        entities = self.normalize_entities(
            entities or []
        )

        # -----------------------------------------------------
        # 2. Add fallback features
        # -----------------------------------------------------
        fallback_features = (
            self.extract_fallback_aspects(
                text,
                entities
            )
        )

        all_entities = entities + fallback_features

        # Sort entities by character position
        all_entities = sorted(
            all_entities,
            key=lambda x: x["start"]
        )

        # -----------------------------------------------------
        # 3. Pair aspects and opinions
        # -----------------------------------------------------
        pairs = self.pair_aspects_and_opinions(
            text,
            all_entities
        )

        # -----------------------------------------------------
        # 4. Calculate aspect sentiment
        # -----------------------------------------------------
        aspect_sentiments = []

        used_pairs = set()

        for pair in pairs:

            aspect = pair["aspect"]
            opinion = pair["opinion"]

            key = (
                aspect["start"],
                aspect["end"],
                opinion["start"],
                opinion["end"]
            )

            # Prevent duplicate aspect-opinion pairs
            if key in used_pairs:
                continue

            used_pairs.add(key)

            sentiment = (
                self.calculate_aspect_sentiment(
                    aspect,
                    opinion
                )
            )

            aspect_sentiments.append({
                "aspect": aspect["text"],
                "aspect_label": aspect["label"],
                "opinion": opinion["text"],
                "opinion_label": opinion["label"],
                "sentiment": sentiment["label"],
                "score": round(
                    sentiment["score"],
                    4
                ),
                "reason": sentiment["reason"],
                "aspect_start": aspect["start"],
                "aspect_end": aspect["end"],
                "opinion_start": opinion["start"],
                "opinion_end": opinion["end"]
            })

        # -----------------------------------------------------
        # 5. Full-review VADER sentiment
        # -----------------------------------------------------
        review_sentiment = (
            self.get_review_sentiment(text)
        )

        # -----------------------------------------------------
        # 6. Overall sentiment
        # -----------------------------------------------------
        overall_sentiment = (
            self.get_overall_sentiment(
                aspect_sentiments,
                review_sentiment
            )
        )

        # -----------------------------------------------------
        # 7. Build user-friendly explanation
        # -----------------------------------------------------
        explanation = (
            overall_sentiment["explanation"]
        )

        if aspect_sentiments:

            aspect_reasons = []

            for item in aspect_sentiments:

                aspect_reasons.append(
                    f"{item['aspect']} is "
                    f"{item['sentiment'].lower()} "
                    f"because of the opinion "
                    f"'{item['opinion']}'."
                )

            explanation += " " + " ".join(
                aspect_reasons
            )

        return {
            "entities": all_entities,

            "aspect_sentiments":
                aspect_sentiments,

            "overall_sentiment":
                overall_sentiment,

            "review_sentiment":
                review_sentiment,

            "explanation":
                explanation
        }