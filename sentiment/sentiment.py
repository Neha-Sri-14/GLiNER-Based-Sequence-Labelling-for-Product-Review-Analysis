import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# Download VADER lexicon if required
# ---------------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class AspectSentimentAnalyzer:

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

        # Product-review aspects observed in the dataset
        self.fallback_aspects = [
            "battery life",
            "battery",
            "build quality",
            "camera quality",
            "camera",
            "screen",
            "display",
            "reception",
            "signal",
            "text messaging",
            "messaging",
            "call quality",
            "calls",
            "network",
            "internet",
            "software",
            "software update",
            "charging",
            "charger",
            "menu",
            "buttons",
            "vibration",
            "ringer",
            "ringtones",
            "voice dialing",
            "speed dialing",
            "games",
            "case",
            "headset",
            "accessories",
            "speaker",
            "speakers",
            "audio",
            "sound",
            "storage",
            "memory",
            "performance",
            "speed",
            "design",
            "quality",
            "price",
            "customer service",
            "service",
            "features",
            "ease of use",
            "usability",
            "durability",
            "size",
            "color",
            "touch",
            "touchscreen",
            "keyboard",
            "trackpad",
            "SIM",
            "SIM card",
            "SD card",
            "reliability",
        ]


    # =====================================================
    # 1. NORMALIZE GLINER ENTITIES
    # =====================================================

    def normalize_entities(self, entities):
        """
        Converts GLiNER output into a consistent format.

        Expected GLiNER output example:

        {
            "text": "battery life",
            "label": "feature",
            "start": 25,
            "end": 37,
            "score": 0.91
        }
        """

        normalized = []

        for entity in entities:

            text = str(entity.get("text", "")).strip()

            if not text:
                continue

            label = str(
                entity.get("label", "")
            ).strip().upper()

            # Keep only the labels used in this project
            allowed_labels = {
                "BRAND",
                "PRODUCT",
                "FEATURE",
                "ISSUE",
                "OPINION"
            }

            if label not in allowed_labels:
                continue

            normalized_entity = {
                "text": text,
                "label": label,
                "start": entity.get("start"),
                "end": entity.get("end"),
                "score": entity.get("score", 0.0)
            }

            normalized.append(normalized_entity)

        return normalized


    # =====================================================
    # 2. SENTENCE SPLITTING
    # =====================================================

    def split_sentences(self, text):
        """
        Split product review into sentences/clauses.

        Amazon reviews often contain:
        '.', '!', '?'
        and sometimes semicolon-separated statements.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+|;\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]


    # =====================================================
    # 3. FALLBACK ASPECT EXTRACTION
    # =====================================================

    def extract_fallback_aspects(self, review):
        """
        Used only when GLiNER does not detect FEATURE entities.

        This is adapted to the actual product-review vocabulary
        observed in the dataset.
        """

        found = []

        review_lower = review.lower()

        # Longer aspects first
        aspects = sorted(
            self.fallback_aspects,
            key=len,
            reverse=True
        )

        for aspect in aspects:

            pattern = r"\b" + re.escape(aspect) + r"\b"

            for match in re.finditer(
                pattern,
                review_lower,
                flags=re.IGNORECASE
            ):

                found.append({
                    "text": review[
                        match.start():match.end()
                    ],
                    "label": "FEATURE",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0
                })

        # Remove overlapping matches
        found.sort(
            key=lambda x: (
                x["start"],
                -(x["end"] - x["start"])
            )
        )

        selected = []

        for entity in found:

            overlap = False

            for existing in selected:

                if (
                    entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    overlap = True
                    break

            if not overlap:
                selected.append(entity)

        return selected


    # =====================================================
    # 4. FIND SENTENCE FOR ENTITY
    # =====================================================

    def get_sentence_for_entity(self, review, entity):

        start = entity.get("start")

        if start is None:
            return review

        sentences = self.split_sentences(review)

        current_position = 0

        for sentence in sentences:

            sentence_start = review.find(
                sentence,
                current_position
            )

            if sentence_start == -1:
                continue

            sentence_end = (
                sentence_start + len(sentence)
            )

            if (
                sentence_start <= start
                <= sentence_end
            ):
                return sentence

            current_position = sentence_end

        return review


    # =====================================================
    # 5. PAIR FEATURES WITH OPINIONS / ISSUES
    # =====================================================

    def pair_aspects_and_opinions(
        self,
        entities,
        review=None
    ):

        features = [
            entity
            for entity in entities
            if entity["label"] == "FEATURE"
        ]

        descriptors = [
            entity
            for entity in entities
            if entity["label"] in {
                "OPINION",
                "ISSUE"
            }
        ]

        pairs = []

        # -------------------------------------------------
        # If no FEATURE was found
        # -------------------------------------------------

        if not features:

            for descriptor in descriptors:

                pairs.append({
                    "feature": "Overall Product",
                    "descriptor": descriptor["text"],
                    "descriptor_type": descriptor["label"],
                    "score": descriptor.get("score", 0.0)
                })

            return pairs


        # -------------------------------------------------
        # Pair every opinion/issue with closest feature
        # -------------------------------------------------

        for descriptor in descriptors:

            desc_start = descriptor.get("start")

            if desc_start is None:

                # Fallback when character offsets unavailable
                closest_feature = features[0]

            else:

                def distance(feature):

                    feature_start = feature.get("start")

                    if feature_start is None:
                        return 999999

                    return abs(
                        feature_start - desc_start
                    )

                closest_feature = min(
                    features,
                    key=distance
                )

            pairs.append({
                "feature": closest_feature["text"],
                "descriptor": descriptor["text"],
                "descriptor_type": descriptor["label"],
                "score": descriptor.get("score", 0.0)
            })

        return pairs


    # =====================================================
    # 6. CALCULATE SENTIMENT
    # =====================================================

    def calculate_aspect_sentiment(
        self,
        pairs,
        review_rating=None
    ):

        results = []

        for pair in pairs:

            feature = pair["feature"]
            descriptor = pair["descriptor"]
            descriptor_type = pair["descriptor_type"]

            # ---------------------------------------------
            # ISSUE = negative
            # ---------------------------------------------

            if descriptor_type == "ISSUE":

                compound = -0.75
                sentiment = "NEGATIVE"

            else:

                scores = self.sia.polarity_scores(
                    descriptor
                )

                compound = scores["compound"]

                if compound >= 0.05:
                    sentiment = "POSITIVE"

                elif compound <= -0.05:
                    sentiment = "NEGATIVE"

                else:
                    sentiment = "NEUTRAL"

            results.append({
                "feature": feature,
                "descriptor": descriptor,
                "descriptor_type": descriptor_type,
                "sentiment": sentiment,
                "score": round(compound, 2)
            })

        return results


    # =====================================================
    # 7. REVIEW-LEVEL SENTIMENT
    # =====================================================

    def get_review_sentiment(self, review):

        scores = self.sia.polarity_scores(review)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "POSITIVE"

        elif compound <= -0.05:
            sentiment = "NEGATIVE"

        else:
            sentiment = "NEUTRAL"

        return sentiment, round(compound, 2)


    # =====================================================
    # 8. RATING-BASED SENTIMENT
    # =====================================================

    def get_rating_sentiment(self, rating):

        if rating is None:
            return None

        try:
            rating = float(rating)
        except (ValueError, TypeError):
            return None

        if rating <= 2:
            return "NEGATIVE"

        elif rating == 3:
            return "NEUTRAL"

        else:
            return "POSITIVE"


    # =====================================================
    # 9. OVERALL SENTIMENT
    # =====================================================

    def get_overall_sentiment(
        self,
        aspect_sentiments,
        review=None,
        review_rating=None
    ):

        # -------------------------------------------------
        # No aspect sentiment
        # -------------------------------------------------

        if not aspect_sentiments:

            if review:

                review_sentiment, compound = (
                    self.get_review_sentiment(review)
                )

                rating_sentiment = (
                    self.get_rating_sentiment(
                        review_rating
                    )
                )

                if rating_sentiment:

                    explanation = (
                        "No specific product feature sentiment "
                        "was extracted. "
                        f"The review text is {review_sentiment.lower()} "
                        f"(VADER score {compound:.2f}), "
                        f"while the customer rating indicates "
                        f"{rating_sentiment.lower()} sentiment."
                    )

                else:

                    explanation = (
                        "No specific product feature sentiment "
                        "was extracted. "
                        f"The review-level sentiment is "
                        f"{review_sentiment.lower()} "
                        f"(VADER score {compound:.2f})."
                    )

                return (
                    review_sentiment,
                    explanation
                )

            return (
                "NEUTRAL",
                "No sentiment information was extracted."
            )


        # -------------------------------------------------
        # Count aspect sentiments
        # -------------------------------------------------

        positive = [
            item
            for item in aspect_sentiments
            if item["sentiment"] == "POSITIVE"
        ]

        negative = [
            item
            for item in aspect_sentiments
            if item["sentiment"] == "NEGATIVE"
        ]

        neutral = [
            item
            for item in aspect_sentiments
            if item["sentiment"] == "NEUTRAL"
        ]


        # -------------------------------------------------
        # Determine overall sentiment
        # -------------------------------------------------

        if len(positive) > len(negative):

            overall = "POSITIVE"

        elif len(negative) > len(positive):

            overall = "NEGATIVE"

        elif len(positive) == 0 and len(negative) == 0:

            overall = "NEUTRAL"

        else:

            overall = "MIXED"


        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------

        positive_features = list(dict.fromkeys(
            item["feature"]
            for item in positive
        ))

        negative_features = list(dict.fromkeys(
            item["feature"]
            for item in negative
        ))

        neutral_features = list(dict.fromkeys(
            item["feature"]
            for item in neutral
        ))


        explanation_parts = []

        if positive_features:

            explanation_parts.append(
                "positive feedback regarding "
                + ", ".join(positive_features)
            )

        if negative_features:

            explanation_parts.append(
                "negative feedback regarding "
                + ", ".join(negative_features)
            )

        if neutral_features:

            explanation_parts.append(
                "neutral feedback regarding "
                + ", ".join(neutral_features)
            )


        if explanation_parts:

            explanation = (
                "The review contains "
                + " and ".join(explanation_parts)
                + "."
            )

        else:

            explanation = (
                "The extracted aspects do not contain "
                "strong sentiment."
            )


        # Add rating information when available
        rating_sentiment = self.get_rating_sentiment(
            review_rating
        )

        if rating_sentiment:

            explanation += (
                f" The customer gave a "
                f"{int(float(review_rating))}-star rating, "
                f"which corresponds to "
                f"{rating_sentiment.lower()} sentiment."
            )


        return overall, explanation


    # =====================================================
    # 10. COMPLETE ANALYSIS
    # =====================================================

    def analyze(
        self,
        review,
        entities=None,
        review_rating=None,
        product_name=None
    ):

        if not review or not review.strip():

            return {
                "product": product_name,
                "entities": [],
                "aspect_pairs": [],
                "aspect_sentiments": [],
                "overall_sentiment": "NEUTRAL",
                "explanation": "Empty review."
            }


        # ---------------------------------------------
        # Normalize GLiNER entities
        # ---------------------------------------------

        entities = entities or []

        entities = self.normalize_entities(
            entities
        )


        # ---------------------------------------------
        # Fallback aspect extraction
        # ---------------------------------------------

        if not any(
            entity["label"] == "FEATURE"
            for entity in entities
        ):

            fallback_features = (
                self.extract_fallback_aspects(review)
            )

            entities.extend(
                fallback_features
            )


        # ---------------------------------------------
        # Remove duplicate entities
        # ---------------------------------------------

        unique_entities = []

        seen = set()

        for entity in entities:

            key = (
                entity["text"].lower(),
                entity["label"],
                entity.get("start"),
                entity.get("end")
            )

            if key not in seen:

                seen.add(key)
                unique_entities.append(entity)


        entities = unique_entities


        # ---------------------------------------------
        # Pair aspects and opinions
        # ---------------------------------------------

        pairs = self.pair_aspects_and_opinions(
            entities,
            review
        )


        # ---------------------------------------------
        # Aspect sentiment
        # ---------------------------------------------

        aspect_sentiments = (
            self.calculate_aspect_sentiment(
                pairs,
                review_rating
            )
        )


        # ---------------------------------------------
        # Overall sentiment
        # ---------------------------------------------

        overall, explanation = (
            self.get_overall_sentiment(
                aspect_sentiments,
                review,
                review_rating
            )
        )


        return {
            "product": product_name,
            "entities": entities,
            "aspect_pairs": pairs,
            "aspect_sentiments": aspect_sentiments,
            "overall_sentiment": overall,
            "explanation": explanation
        }