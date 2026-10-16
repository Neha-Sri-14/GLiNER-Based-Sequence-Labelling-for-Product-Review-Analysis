import json
import os
import re

from gliner import GLiNER

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)


# =========================================================
# 1. CONVERT TOKEN-INDEX NER SPANS TO BIO
# =========================================================

def ner_to_bio(tokens, ner_spans):
    """
    Converts GLiNER token-index spans into BIO labels.

    Input:
        tokens = [
            "The",
            "camera",
            "quality",
            "is",
            "excellent"
        ]

        ner_spans = [
            [1, 2, "feature"],
            [4, 4, "opinion"]
        ]

    Output:
        [
            "O",
            "B-FEATURE",
            "I-FEATURE",
            "O",
            "B-OPINION"
        ]
    """

    bio_tags = ["O"] * len(tokens)

    # -----------------------------------------------------
    # Sort spans
    # -----------------------------------------------------

    valid_spans = []

    for span in ner_spans:

        if len(span) != 3:
            continue

        start, end, label = span

        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue

        label = str(label).strip().upper()

        # Validity checks
        if start < 0:
            continue

        if end < start:
            continue

        if start >= len(tokens):
            continue

        end = min(end, len(tokens) - 1)

        valid_spans.append(
            [start, end, label]
        )


    # -----------------------------------------------------
    # Sort by starting token
    # -----------------------------------------------------

    valid_spans.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )


    # -----------------------------------------------------
    # Assign BIO labels
    #
    # Do not allow overlapping entities.
    # -----------------------------------------------------

    occupied = set()

    for start, end, label in valid_spans:

        token_indices = range(
            start,
            end + 1
        )

        # Skip an entity if it overlaps a previously
        # assigned entity.
        if any(
            index in occupied
            for index in token_indices
        ):
            continue


        # B-label
        bio_tags[start] = (
            f"B-{label}"
        )

        occupied.add(start)


        # I-label
        for index in range(
            start + 1,
            end + 1
        ):

            bio_tags[index] = (
                f"I-{label}"
            )

            occupied.add(index)


    return bio_tags


# =========================================================
# 2. CREATE TEXT FROM DATASET TOKENS
# =========================================================

def reconstruct_text_with_offsets(tokens):
    """
    Reconstructs the text from dataset tokens while
    preserving the exact character offsets used for
    prediction alignment.

    Returns:
        raw_text
        token_offsets

    Example:

        raw_text = "The camera quality is excellent."

        token_offsets = [
            (0, 3),
            (4, 10),
            (11, 18),
            ...
        ]
    """

    raw_text_parts = []
    token_offsets = []

    current_position = 0

    for token in tokens:

        token = str(token)

        # Add a single space between dataset tokens
        if raw_text_parts:

            current_position += 1

        start = current_position

        raw_text_parts.append(token)

        end = start + len(token)

        token_offsets.append(
            (start, end)
        )

        current_position = end


    raw_text = " ".join(
        raw_text_parts
    )

    return raw_text, token_offsets


# =========================================================
# 3. CONVERT GLINER CHARACTER SPANS TO TOKEN SPANS
# =========================================================

def gliner_to_token_spans(
    predictions,
    token_offsets
):
    """
    Converts GLiNER character-level predictions into
    token-index spans.

    GLiNER:
        start/end = character offsets

    Dataset:
        start/end = token indices

    Output:
        [
            [start_token, end_token, label]
        ]
    """

    token_spans = []

    for prediction in predictions:

        try:
            p_start = int(
                prediction["start"]
            )

            p_end = int(
                prediction["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            continue


        label = str(
            prediction.get("label", "")
        ).strip().upper()


        if not label:
            continue


        start_token = None
        end_token = None


        # -------------------------------------------------
        # Find all tokens overlapping the GLiNER span
        # -------------------------------------------------

        overlapping_tokens = []

        for index, (
            token_start,
            token_end
        ) in enumerate(token_offsets):

            # Character-span overlap
            if (
                token_start < p_end
                and token_end > p_start
            ):

                overlapping_tokens.append(
                    index
                )


        if not overlapping_tokens:
            continue


        start_token = overlapping_tokens[0]
        end_token = overlapping_tokens[-1]


        if start_token <= end_token:

            token_spans.append([
                start_token,
                end_token,
                label
            ])


    return token_spans


# =========================================================
# 4. REMOVE OVERLAPPING PREDICTIONS
# =========================================================

def remove_overlapping_spans(spans):
    """
    BIO sequence labelling can assign only one label to
    each token.

    Therefore, overlapping GLiNER predictions must be
    resolved before generating BIO tags.
    """

    if not spans:
        return []


    # Prefer earlier spans and, for equal starts,
    # prefer the longer span.
    spans = sorted(
        spans,
        key=lambda x: (
            x[0],
            -(x[1] - x[0])
        )
    )


    selected = []
    occupied = set()


    for start, end, label in spans:

        current_tokens = set(
            range(
                start,
                end + 1
            )
        )


        # Skip overlapping span
        if occupied.intersection(
            current_tokens
        ):
            continue


        selected.append([
            start,
            end,
            label
        ])

        occupied.update(
            current_tokens
        )


    return selected


# =========================================================
# 5. EVALUATE TEST SET
# =========================================================

def evaluate_on_test_set(
    test_json_path,
    model_path=None
):

    print("=" * 70)
    print("GLiNER SEQUENCE LABELLING EVALUATION")
    print("=" * 70)


    # -----------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------

    print(
        f"\nLoading test dataset from:\n"
        f"{test_json_path}"
    )


    with open(
        test_json_path,
        "r",
        encoding="utf-8"
    ) as f:

        test_data = json.load(f)


    print(
        f"Test instances: {len(test_data)}"
    )


    # -----------------------------------------------------
    # Model path
    # -----------------------------------------------------

    if model_path is None:

        # Project structure:
        #
        # ProductReview_GLiner/
        # ├── dataset/
        # │   └── test_clean.json
        # ├── gliner_model/
        # │   └── model/
        # └── evaluation/
        #     └── evaluate.py

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        model_path = os.path.join(
            project_root,
            "gliner_model",
            "model"
        )


    print(
        f"\nLoading model from:\n"
        f"{model_path}"
    )


    # -----------------------------------------------------
    # Load GLiNER
    # -----------------------------------------------------

    model = GLiNER.from_pretrained(
        model_path,
        load_tokenizer=True
    )


    # -----------------------------------------------------
    # Entity labels used by your dataset
    # -----------------------------------------------------

    labels = [
        "brand",
        "product",
        "feature",
        "issue",
        "opinion"
    ]


    # -----------------------------------------------------
    # Storage for seqeval
    # -----------------------------------------------------

    true_sequences = []
    pred_sequences = []


    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    print(
        "\nRunning inference..."
    )


    for idx, sample in enumerate(
        test_data
    ):

        tokens = sample[
            "tokenized_text"
        ]

        gold_ner = sample[
            "ner"
        ]


        # =============================================
        # GOLD BIO
        # =============================================

        gold_bio = ner_to_bio(
            tokens,
            gold_ner
        )


        # Safety check
        if len(gold_bio) != len(tokens):

            print(
                f"WARNING: Gold BIO length mismatch "
                f"at sample {idx}"
            )

            continue


        # =============================================
        # RECONSTRUCT TEXT
        # =============================================

        raw_text, token_offsets = (
            reconstruct_text_with_offsets(
                tokens
            )
        )


        # =============================================
        # GLINER PREDICTION
        # =============================================

        try:

            predictions = (
                model.predict_entities(
                    raw_text,
                    labels,
                    threshold=0.5
                )
            )

        except Exception as e:

            print(
                f"WARNING: Prediction failed "
                f"at sample {idx}: {e}"
            )

            predictions = []


        # =============================================
        # CHARACTER SPANS → TOKEN SPANS
        # =============================================

        pred_ner = gliner_to_token_spans(
            predictions,
            token_offsets
        )


        # =============================================
        # REMOVE OVERLAPPING PREDICTIONS
        # =============================================

        pred_ner = remove_overlapping_spans(
            pred_ner
        )


        # =============================================
        # PREDICTED BIO
        # =============================================

        pred_bio = ner_to_bio(
            tokens,
            pred_ner
        )


        # Safety check
        if len(pred_bio) != len(tokens):

            print(
                f"WARNING: Prediction BIO length "
                f"mismatch at sample {idx}"
            )

            continue


        # =============================================
        # SAVE SEQUENCES
        # =============================================

        true_sequences.append(
            gold_bio
        )

        pred_sequences.append(
            pred_bio
        )


        # Progress
        if (
            (idx + 1) % 50 == 0
            or idx == 0
            or idx == len(test_data) - 1
        ):

            print(
                f"Processed "
                f"{idx + 1}/{len(test_data)}"
            )


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEQUENCE LABELLING RESULTS"
    )

    print(
        "=" * 70
    )


    if not true_sequences:

        print(
            "No valid test sequences were evaluated."
        )

        return


    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            true_sequences,
            pred_sequences,
            digits=4
        )
    )


    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    overall_f1 = f1_score(
        true_sequences,
        pred_sequences
    )

    overall_precision = precision_score(
        true_sequences,
        pred_sequences
    )

    overall_recall = recall_score(
        true_sequences,
        pred_sequences
    )


    print(
        "-" * 70
    )

    print(
        f"Overall Entity F1-Score : "
        f"{overall_f1:.4f}"
    )

    print(
        f"Overall Precision       : "
        f"{overall_precision:.4f}"
    )

    print(
        f"Overall Recall          : "
        f"{overall_recall:.4f}"
    )

    print(
        "=" * 70
    )


# =========================================================
# 6. MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Use CLEANED test dataset
    # -----------------------------------------------------

    test_file = os.path.join(
        "..",
        "dataset",
        "test_clean.json"
    )


    # -----------------------------------------------------
    # Optional:
    # specify your trained model directory here.
    #
    # Example:
    #
    # model_directory = os.path.join(
    #     "..",
    #     "gliner_model",
    #     "model"
    # )
    #
    # -----------------------------------------------------

    model_directory = None


    evaluate_on_test_set(
        test_file,
        model_directory
    )