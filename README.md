# GLiNER-Based Sequence Labelling for Product Review Analysis

A Flask web application for analyzing product reviews using GLiNER-based
entity extraction, BIO sequence labelling, aspect--opinion pairing, and
sentiment analysis.

## Overview

The project extracts useful information from customer reviews to
identify product features, issues, opinions, and related sentiment.

### Main Features

-   Extract `BRAND`, `PRODUCT`, `FEATURE`, `ISSUE`, and `OPINION`
    entities.
-   Convert GLiNER entity spans into BIO token labels (`B-`, `I-`, `O`).
-   Pair product features with related opinions or issues where
    possible.
-   Estimate aspect-level and overall sentiment.
-   Display product analysis through a Flask web interface.
-   Evaluate extraction using precision, recall, F1-score, and a
    classification report.

> Sentiment and entity outputs are estimates and may require manual
> review.

## Pipeline

``` text
Product
  ↓
User Review
  ↓
Preprocessing
  ↓
GLiNER Entity Extraction
  ↓
Span-to-BIO Conversion
  ↓
Sequence Labelling
  ↓
Brand / Product / Feature / Issue / Opinion
  ↓
Aspect–Opinion Pairing
  ↓
Sentiment Analysis
  ↓
Flask Web Interface
```

## Technology Stack

-   Python
-   GLiNER (`urchade/gliner_small-v2.1` as the development base model)
-   PyTorch
-   Flask
-   Pandas / NumPy, where used
-   NLTK VADER, where configured for sentiment
-   seqeval, for sequence-labelling evaluation
-   HTML, CSS, and JavaScript for the frontend, if included

## Project Structure

Adapt this example to match the actual filenames in your repository:

``` text
ProductReview_GLiner/
├── app.py
├── ner_predictor.py
├── dataset/
│   ├── train_clean.json
│   ├── validation_clean.json
│   └── test_clean.json
├── gliner_model/
│   └── model/
├── evaluation/
│   ├── evaluate.py
│   └── validate_dataset.py
├── templates/
├── static/
├── requirements.txt
└── README.md
```

## Dataset Format

Training and evaluation JSON records use tokenized text and entity
spans. Example:

``` json
{
  "tokenized_text": ["The", "Samsung", "Galaxy", "has", "an", "excellent", "camera", "."],
  "ner": [
    [1, 1, "brand"],
    [2, 2, "product"],
    [6, 6, "feature"],
    [5, 5, "opinion"]
  ]
}
```

Each entity has the form `[start_token_index, end_token_index, label]`.
Ensure that the dataset and training code use the same indexing
convention.

### BIO Labels

  Token       BIO label
  ----------- -------------
  The         `O`
  camera      `B-FEATURE`
  quality     `I-FEATURE`
  is          `O`
  excellent   `B-OPINION`

-   `B-LABEL`: beginning of an entity
-   `I-LABEL`: continuation of an entity
-   `O`: token outside a labelled entity

## Requirements

-   Python version supported by the installed GLiNER and PyTorch
    versions
-   `pip`
-   Internet access to install packages/download the base model, unless
    already available locally
-   The fine-tuned checkpoint, if the application uses a trained model

## Installation (Windows / VS Code)

Open a terminal in the project directory.

### 1. Create and activate a virtual environment

PowerShell:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Command Prompt alternative:

``` bat
venv\Scripts\activate.bat
```

### 2. Install dependencies

If `requirements.txt` exists:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install the packages used by your implementation:

``` bash
pip install Flask gliner torch pandas numpy nltk scikit-learn seqeval
pip freeze > requirements.txt
```

Keep only dependencies actually required by the project.

### 3. Check the model path

Confirm that the fine-tuned model files exist and that `app.py` or
`ner_predictor.py` loads the correct directory. The base model ID does
not automatically point to your fine-tuned checkpoint.

## Run the Application

From the project root, activate the environment and run:

``` bash
python app.py
```

Open the local URL printed in the terminal (commonly
`http://127.0.0.1:5000/`). If your application uses another entry point
or port, follow its configuration.

## Training

If your training script is included and the cleaned dataset files are
ready, run the appropriate script from your project. The development
configuration included:

-   Base model: `urchade/gliner_small-v2.1`
-   Learning rate: `2e-5`
-   Per-device training batch size: `8`
-   Maximum steps: `2000`
-   Epoch configuration: `1`
-   Weight decay: `0.01`

These are configuration values, not measured performance guarantees.
Save the resulting checkpoint and ensure the application loads it for
inference.

## Dataset Validation and Evaluation

If these scripts exist at the listed paths, run them from the project
root:

``` bash
python evaluation/validate_dataset.py
python evaluation/evaluate.py
```

Check each script for its expected paths and arguments. Evaluation
metrics include:

-   **Precision:** proportion of predicted entities that are correct.
-   **Recall:** proportion of reference entities recovered.
-   **F1-score:** harmonic mean of precision and recall.
-   **Classification report:** per-label evaluation results.

Use the held-out test set for final results. Do not publish performance
scores unless you have measured them.

## Entity Labels

  Label       Meaning                         Example
  ----------- ------------------------------- ------------------------
  `brand`     Brand name                      Samsung
  `product`   Product name/model              Galaxy
  `feature`   Product aspect                  camera quality
  `issue`     Problem or negative condition   battery drains quickly
  `opinion`   Descriptive opinion             excellent, beautiful

## Deployment Notes

For a platform such as Render:

1.  Push the project to a Git repository.
2.  Include a correct `requirements.txt`.
3.  Configure build and start commands for the actual Flask application.
4.  Add required environment variables in the hosting dashboard.
5.  Ensure the model checkpoint is accessible during deployment.
6.  Check model size, memory requirements, build limits, and request
    timeouts.

Do not commit API keys, credentials, `.env` files, virtual environments,
caches, or other secrets. Consider storing large model checkpoints and
datasets separately if repository or hosting limits require it.

Example `.gitignore` entries (adapt as needed):

``` gitignore
venv/
.venv/
__pycache__/
*.py[cod]
.env
.pytest_cache/
.ipynb_checkpoints/
```

## Limitations

-   GLiNER extracts entity spans; this project converts those spans into
    BIO labels.
-   Results depend on dataset annotation quality, label definitions, and
    the model checkpoint.
-   Aspect--opinion pairing may be imperfect for complex reviews.
-   Rule-based sentiment analysis may miss sarcasm, negation, or
    domain-specific meaning.
-   Product images and metadata depend on the application's configured
    data sources.

## Future Enhancements

-   Manually review and improve entity annotations.
-   Test punctuation-aware tokenization and character-offset alignment.
-   Evaluate performance per entity label.
-   Improve aspect--opinion pairing for reviews discussing multiple
    features.
-   Add product search, filtering, and pagination for larger catalogs.
-   Add automated tests, logging, and deployment monitoring.

## Project Information

**Title:** GLiNER-Based Sequence Labelling for Product Review Analysis

Before publishing, update the project structure, commands, repository
URL, author details, and evaluation results to match the final
implementation.
