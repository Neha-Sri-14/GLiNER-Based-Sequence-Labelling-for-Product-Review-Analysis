import json
import os


# =========================================================
# CONFIGURATION
# =========================================================

ALLOWED_LABELS = {
    "brand",
    "product",
    "feature",
    "issue",
    "opinion"
}


# =========================================================
# VALIDATE ONE DATASET SPLIT
# =========================================================

def validate_split(file_path):

    filename = os.path.basename(file_path)

    # -----------------------------------------------------
    # Check file existence
    # -----------------------------------------------------

    if not os.path.exists(file_path):

        print(
            f"[-] Missing file: {filename}"
        )

        return False


    # -----------------------------------------------------
    # Load JSON
    # -----------------------------------------------------

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except json.JSONDecodeError as e:

        print(
            f"[-] {filename}: Invalid JSON"
        )

        print(
            f"    Error: {e}"
        )

        return False


    # -----------------------------------------------------
    # Root structure
    # -----------------------------------------------------

    if not isinstance(data, list):

        print(
            f"[-] {filename}: "
            f"Root JSON must be a list."
        )

        return False


    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    error_count = 0

    warning_count = 0

    total_entities = 0

    total_tokens = 0

    label_counts = {
        label.upper(): 0
        for label in ALLOWED_LABELS
    }


    # =====================================================
    # CHECK EACH RECORD
    # =====================================================

    for record_index, record in enumerate(data):

        # -------------------------------------------------
        # Record must be dictionary
        # -------------------------------------------------

        if not isinstance(record, dict):

            print(
                f"[-] {filename} record "
                f"{record_index}: "
                f"Record must be a JSON object."
            )

            error_count += 1

            continue


        # -------------------------------------------------
        # Check required fields
        # -------------------------------------------------

        if "tokenized_text" not in record:

            print(
                f"[-] {filename} record "
                f"{record_index}: "
                f"Missing 'tokenized_text'."
            )

            error_count += 1

            continue


        if "ner" not in record:

            print(
                f"[-] {filename} record "
                f"{record_index}: "
                f"Missing 'ner'."
            )

            error_count += 1

            continue


        # -------------------------------------------------
        # Get tokens and entities
        # -------------------------------------------------

        tokens = record["tokenized_text"]

        entities = record["ner"]


        # -------------------------------------------------
        # tokenized_text must be list
        # -------------------------------------------------

        if not isinstance(tokens, list):

            print(
                f"[-] {filename} record "
                f"{record_index}: "
                f"'tokenized_text' must be a list."
            )

            error_count += 1

            continue


        # -------------------------------------------------
        # ner must be list
        # -------------------------------------------------

        if not isinstance(entities, list):

            print(
                f"[-] {filename} record "
                f"{record_index}: "
                f"'ner' must be a list."
            )

            error_count += 1

            continue


        # -------------------------------------------------
        # Check empty token list
        # -------------------------------------------------

        if len(tokens) == 0:

            print(
                f"[!] {filename} record "
                f"{record_index}: "
                f"Empty tokenized_text."
            )

            warning_count += 1


        total_tokens += len(tokens)

        total_entities += len(entities)


        max_idx = len(tokens) - 1


        # -------------------------------------------------
        # Track occupied token indices
        #
        # Used to detect overlapping entities.
        # -------------------------------------------------

        occupied_tokens = {}


        # =================================================
        # CHECK ENTITIES
        # =================================================

        for entity_index, entity in enumerate(
            entities
        ):

            # -------------------------------------------------
            # Entity must be list
            # -------------------------------------------------

            if not isinstance(entity, list):

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"Entity must be a list."
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Entity must contain:
            #
            # [start, end, label]
            # -------------------------------------------------

            if len(entity) != 3:

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"Expected [start, end, label], "
                    f"got {entity}."
                )

                error_count += 1

                continue


            start, end, label = entity


            # -------------------------------------------------
            # Validate start/end type
            # -------------------------------------------------

            if not isinstance(start, int):

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"Start index must be integer."
                )

                error_count += 1

                continue


            if not isinstance(end, int):

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"End index must be integer."
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Validate label
            # -------------------------------------------------

            if not isinstance(label, str):

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"Label must be a string."
                )

                error_count += 1

                continue


            label_lower = label.strip().lower()


            if label_lower not in ALLOWED_LABELS:

                print(
                    f"[-] {filename} record "
                    f"{record_index}, entity "
                    f"{entity_index}: "
                    f"Unknown label '{label}'. "
                    f"Allowed labels: "
                    f"{sorted(ALLOWED_LABELS)}"
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Check start <= end
            # -------------------------------------------------

            if start > end:

                print(
                    f"[-] {filename} record "
                    f"{record_index}: "
                    f"Entity '{label}' "
                    f"[{start}, {end}] has "
                    f"start > end."
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Check negative indices
            # -------------------------------------------------

            if start < 0:

                print(
                    f"[-] {filename} record "
                    f"{record_index}: "
                    f"Entity '{label}' "
                    f"[{start}, {end}] has "
                    f"negative start index."
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Check end index
            # -------------------------------------------------

            if end > max_idx:

                print(
                    f"[-] {filename} record "
                    f"{record_index}: "
                    f"Entity '{label}' "
                    f"[{start}, {end}] exceeds "
                    f"token length {len(tokens)}."
                )

                error_count += 1

                continue


            # -------------------------------------------------
            # Check empty token text
            # -------------------------------------------------

            entity_tokens = tokens[
                start:end + 1
            ]


            if any(
                not isinstance(token, str)
                or not token.strip()
                for token in entity_tokens
            ):

                print(
                    f"[-] {filename} record "
                    f"{record_index}: "
                    f"Entity '{label}' "
                    f"[{start}, {end}] "
                    f"contains an empty token."
                )

                error_count += 1


            # -------------------------------------------------
            # Check overlapping entities
            # -------------------------------------------------

            overlap_found = False

            for token_index in range(
                start,
                end + 1
            ):

                if token_index in occupied_tokens:

                    previous_entity = (
                        occupied_tokens[
                            token_index
                        ]
                    )

                    print(
                        f"[!] {filename} record "
                        f"{record_index}: "
                        f"Overlapping entities at "
                        f"token {token_index}: "
                        f"'{previous_entity}' "
                        f"and '{label}'."
                    )

                    warning_count += 1

                    overlap_found = True


            # -------------------------------------------------
            # Mark tokens as occupied
            # -------------------------------------------------

            for token_index in range(
                start,
                end + 1
            ):

                occupied_tokens[
                    token_index
                ] = label


            # -------------------------------------------------
            # Count label
            # -------------------------------------------------

            label_counts[
                label_lower.upper()
            ] += 1


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 65)

    print(
        f"DATASET: {filename}"
    )

    print("=" * 65)


    print(
        f"Samples             : {len(data)}"
    )

    print(
        f"Total tokens        : {total_tokens}"
    )

    print(
        f"Total entities      : {total_entities}"
    )

    print(
        f"Errors              : {error_count}"
    )

    print(
        f"Warnings            : {warning_count}"
    )


    # -----------------------------------------------------
    # Label distribution
    # -----------------------------------------------------

    print(
        "\nEntity Label Distribution:"
    )

    for label, count in sorted(
        label_counts.items()
    ):

        print(
            f"  {label:<10}: {count}"
        )


    # =====================================================
    # FINAL STATUS
    # =====================================================

    if error_count == 0:

        print(
            f"\n[+] {filename}: "
            f"PASSED validation."
        )

        if warning_count > 0:

            print(
                f"[!] {filename}: "
                f"Passed with {warning_count} "
                f"warning(s)."
            )

        return True

    else:

        print(
            f"\n[-] {filename}: "
            f"FAILED validation with "
            f"{error_count} error(s)."
        )

        return False


# =========================================================
# VALIDATE ALL DATASETS
# =========================================================

def validate_all_datasets(
    dataset_dir="../dataset"
):

    # -----------------------------------------------------
    # Use cleaned dataset files
    # -----------------------------------------------------

    splits = [
        "train_clean.json",
        "validation_clean.json",
        "test_clean.json"
    ]


    all_passed = True


    print(
        "\n"
        + "=" * 65
    )

    print(
        "PRODUCT REVIEW GLINER DATASET VALIDATION"
    )

    print(
        "=" * 65
    )


    for split in splits:

        path = os.path.join(
            dataset_dir,
            split
        )


        status = validate_split(
            path
        )


        if not status:

            all_passed = False


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FINAL DATASET VALIDATION"
    )

    print(
        "=" * 65
    )


    if all_passed:

        print(
            "[+] All cleaned dataset splits "
            "passed validation."
        )

    else:

        print(
            "[-] One or more dataset splits "
            "contain errors."
        )


    return all_passed


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    validate_all_datasets()