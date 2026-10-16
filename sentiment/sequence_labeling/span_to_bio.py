import re


# =========================================================
# 1. TOKENIZATION WITH CHARACTER OFFSETS
# =========================================================

def tokenize_with_offsets(text):
    """
    Tokenizes the original review while preserving
    character offsets.

    Whitespace tokenization is intentionally used here
    because the GLiNER character offsets are based on
    the original review text.

    Returns:
        [
            {
                "token": "camera",
                "start": 4,
                "end": 10
            },
            ...
        ]
    """

    tokens = []

    for match in re.finditer(r"\S+", text):

        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


# =========================================================
# 2. NORMALIZE GLINER LABEL
# =========================================================

def normalize_label(label):
    """
    Converts GLiNER labels into the labels used
    throughout this project.

    Example:
        "feature" -> "FEATURE"
        "Product" -> "PRODUCT"
    """

    if not label:
        return None

    label = str(label).strip().upper()

    allowed_labels = {
        "BRAND",
        "PRODUCT",
        "FEATURE",
        "ISSUE",
        "OPINION"
    }

    if label not in allowed_labels:
        return None

    return label


# =========================================================
# 3. CHECK WHETHER TWO SPANS OVERLAP
# =========================================================

def spans_overlap(start1, end1, start2, end2):
    """
    Returns True if two character spans overlap.

    Character spans follow Python's convention:

        start = inclusive
        end   = exclusive
    """

    return (
        start1 < end2
        and end1 > start2
    )


# =========================================================
# 4. CONVERT GLINER SPANS TO BIO
# =========================================================

def convert_spans_to_bio(text, gliner_entities):
    """
    Converts GLiNER character-level entities into
    token-level BIO sequence labels.

    Args:
        text:
            Original review text.

        gliner_entities:
            Output from:

                model.predict_entities()

            Example:

                {
                    "start": 4,
                    "end": 18,
                    "text": "camera quality",
                    "label": "feature",
                    "score": 0.91
                }

    Returns:
        Dictionary containing:

        {
            "bio_sequence": [
                ("The", "O"),
                ("camera", "B-FEATURE"),
                ("quality", "I-FEATURE"),
                ...
            ],

            "entities": [
                {
                    "text": "camera quality",
                    "label": "FEATURE",
                    "start": 4,
                    "end": 18,
                    "token_start": 1,
                    "token_end": 2,
                    "score": 0.91
                }
            ]
        }
    """

    # -----------------------------------------------------
    # Tokenize original text
    # -----------------------------------------------------

    tokens = tokenize_with_offsets(text)

    if not tokens:
        return {
            "bio_sequence": [],
            "entities": []
        }


    # -----------------------------------------------------
    # Initialize every token as O
    # -----------------------------------------------------

    for token in tokens:
        token["label"] = "O"
        token["entity_id"] = None


    # -----------------------------------------------------
    # Clean GLiNER entities
    # -----------------------------------------------------

    valid_entities = []

    for entity in gliner_entities or []:

        try:
            ent_start = int(entity["start"])
            ent_end = int(entity["end"])

        except (KeyError, TypeError, ValueError):

            continue


        # ---------------------------------------------
        # Invalid character span
        # ---------------------------------------------

        if ent_start < 0:
            continue

        if ent_end <= ent_start:
            continue

        if ent_start >= len(text):
            continue


        # ---------------------------------------------
        # Clamp end position if necessary
        # ---------------------------------------------

        ent_end = min(
            ent_end,
            len(text)
        )


        # ---------------------------------------------
        # Normalize label
        # ---------------------------------------------

        ent_label = normalize_label(
            entity.get("label")
        )

        if ent_label is None:
            continue


        # ---------------------------------------------
        # Store cleaned entity
        # ---------------------------------------------

        valid_entities.append({
            "start": ent_start,
            "end": ent_end,
            "text": text[ent_start:ent_end],
            "label": ent_label,
            "score": float(
                entity.get("score", 0.0)
            )
        })


    # -----------------------------------------------------
    # Sort entities
    #
    # Higher score is preferred when entities start
    # at the same location.
    # -----------------------------------------------------

    valid_entities.sort(
        key=lambda entity: (
            entity["start"],
            -entity["score"],
            -(entity["end"] - entity["start"])
        )
    )


    # -----------------------------------------------------
    # Resolve overlapping entities
    #
    # BIO can represent only one label per token.
    # Therefore overlapping GLiNER spans must be resolved.
    # -----------------------------------------------------

    selected_entities = []

    occupied_token_indices = set()

    for entity in valid_entities:

        overlapping_tokens = []

        for token_index, token in enumerate(tokens):

            if spans_overlap(
                token["start"],
                token["end"],
                entity["start"],
                entity["end"]
            ):
                overlapping_tokens.append(
                    token_index
                )


        if not overlapping_tokens:
            continue


        # ---------------------------------------------
        # Skip entity if it overlaps an already
        # selected entity.
        # ---------------------------------------------

        if any(
            index in occupied_token_indices
            for index in overlapping_tokens
        ):
            continue


        # ---------------------------------------------
        # Save entity
        # ---------------------------------------------

        entity["token_start"] = (
            overlapping_tokens[0]
        )

        entity["token_end"] = (
            overlapping_tokens[-1]
        )

        selected_entities.append(entity)

        occupied_token_indices.update(
            overlapping_tokens
        )


    # -----------------------------------------------------
    # Assign BIO tags
    # -----------------------------------------------------

    for entity in selected_entities:

        token_start = entity["token_start"]
        token_end = entity["token_end"]
        label = entity["label"]

        for index in range(
            token_start,
            token_end + 1
        ):

            if index == token_start:

                tokens[index]["label"] = (
                    f"B-{label}"
                )

            else:

                tokens[index]["label"] = (
                    f"I-{label}"
                )


    # -----------------------------------------------------
    # Create BIO sequence
    # -----------------------------------------------------

    bio_sequence = [
        (
            token["token"],
            token["label"]
        )
        for token in tokens
    ]


    # -----------------------------------------------------
    # Return both BIO and entity information
    # -----------------------------------------------------

    return {
        "bio_sequence": bio_sequence,
        "entities": selected_entities
    }


# =========================================================
# 5. TEST
# =========================================================

if __name__ == "__main__":

    sample_text = (
        "The camera quality is excellent but "
        "the battery drains fast."
    )


    sample_gliner_output = [

        {
            "start": 4,
            "end": 18,
            "text": "camera quality",
            "label": "feature",
            "score": 0.95
        },

        {
            "start": 22,
            "end": 31,
            "text": "excellent",
            "label": "opinion",
            "score": 0.94
        },

        {
            "start": 40,
            "end": 47,
            "text": "battery",
            "label": "feature",
            "score": 0.93
        },

        {
            "start": 56,
            "end": 59,
            "text": "fast",
            "label": "opinion",
            "score": 0.91
        }
    ]


    # -----------------------------------------------------
    # Convert
    # -----------------------------------------------------

    result = convert_spans_to_bio(
        sample_text,
        sample_gliner_output
    )


    # -----------------------------------------------------
    # Display BIO sequence
    # -----------------------------------------------------

    print("\n--- BIO Sequence Output ---")

    for token, tag in result["bio_sequence"]:

        print(
            f"{token:<15} {tag}"
        )


    # -----------------------------------------------------
    # Display entities
    # -----------------------------------------------------

    print("\n--- Extracted Entities ---")

    for entity in result["entities"]:

        print(
            f"{entity['label']:<10} -> "
            f"{entity['text']:<20} "
            f"tokens=({entity['token_start']}, "
            f"{entity['token_end']}) "
            f"chars=({entity['start']}, "
            f"{entity['end']}) "
            f"score={entity['score']:.2f}"
        )