def convert_bio_to_entities(bio_sequence):
    """
    Converts a BIO sequence into complete entity dictionaries.

    Args:
        bio_sequence:
            List of tuples in the form:
            [
                ("The", "O"),
                ("camera", "B-FEATURE"),
                ("quality", "I-FEATURE"),
                ("is", "O"),
                ("excellent", "B-OPINION")
            ]

    Returns:
        list of dicts:
        [
            {
                "text": "camera quality",
                "label": "FEATURE",
                "start": 4,
                "end": 18,
                "token_start": 1,
                "token_end": 2
            }
        ]

    The start/end values are character positions in the
    reconstructed tokenized text.
    """

    entities = []

    current_tokens = []
    current_label = None
    current_token_start = None

    # Character position in reconstructed text
    current_char_start = None
    current_char_position = 0

    def save_current_entity(end_position):
        """Save the current entity before starting another one."""

        nonlocal current_tokens
        nonlocal current_label
        nonlocal current_token_start
        nonlocal current_char_start

        if current_tokens and current_label is not None:

            entity_text = " ".join(current_tokens)

            entities.append({
                "text": entity_text,
                "label": current_label,
                "start": current_char_start,
                "end": end_position,
                "token_start": current_token_start,
                "token_end": current_token_start + len(current_tokens) - 1
            })

        current_tokens = []
        current_label = None
        current_token_start = None
        current_char_start = None


    # -----------------------------------------------------
    # Process BIO sequence
    # -----------------------------------------------------

    for token_index, (token, tag) in enumerate(bio_sequence):

        token = str(token)
        tag = str(tag).upper().strip()

        token_start_position = current_char_position
        token_end_position = (
            token_start_position + len(token)
        )

        # -------------------------------------------------
        # O = Outside
        # -------------------------------------------------

        if tag == "O":

            save_current_entity(
                token_start_position - 1
            )


        # -------------------------------------------------
        # B- = Beginning of entity
        # -------------------------------------------------

        elif tag.startswith("B-"):

            # Save previous entity
            save_current_entity(
                token_start_position - 1
            )

            current_tokens = [token]

            current_label = tag[2:].upper()

            current_token_start = token_index

            current_char_start = token_start_position


        # -------------------------------------------------
        # I- = Inside entity
        # -------------------------------------------------

        elif tag.startswith("I-"):

            label = tag[2:].upper()

            # Correct I- continuation
            if (
                current_tokens
                and current_label == label
            ):

                current_tokens.append(token)

            # Malformed BIO:
            # I- appears without matching B-
            else:

                # Save previous entity
                save_current_entity(
                    token_start_position - 1
                )

                # Treat malformed I- as B-
                current_tokens = [token]

                current_label = label

                current_token_start = token_index

                current_char_start = token_start_position


        # -------------------------------------------------
        # Unknown tag
        # -------------------------------------------------

        else:

            # Treat unknown tags as O
            save_current_entity(
                token_start_position - 1
            )


        # -------------------------------------------------
        # Update character position
        # -------------------------------------------------

        current_char_position = token_end_position + 1


    # -----------------------------------------------------
    # Save final entity
    # -----------------------------------------------------

    save_current_entity(
        current_char_position - 2
    )

    return entities


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample_bio = [
        ("The", "O"),
        ("camera", "B-FEATURE"),
        ("quality", "I-FEATURE"),
        ("is", "O"),
        ("excellent", "B-OPINION"),
        ("but", "O"),
        ("battery", "B-FEATURE"),
        ("life", "I-FEATURE"),
        ("is", "O"),
        ("terrible", "B-ISSUE")
    ]

    print("\n--- Extracted Entities ---")

    extracted = convert_bio_to_entities(
        sample_bio
    )

    for entity in extracted:

        print(
            f"{entity['label']:<10} -> "
            f"{entity['text']:<20} "
            f"chars=({entity['start']}, {entity['end']}) "
            f"tokens=({entity['token_start']}, "
            f"{entity['token_end']})"
        )