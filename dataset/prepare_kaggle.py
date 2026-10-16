import pandas as pd
import json
import re
import random
from gliner import GLiNER

def tokenize_with_offsets(text):
    """Tokenizes text by whitespace and returns tokens with char offsets."""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end()
        })
    return tokens

def auto_annotate_and_split(csv_path, total_samples=1000):
    print(f"Loading merged dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for reviews between 10 and 60 words for optimal NER training
    df['text_len'] = df['review_text'].apply(lambda x: len(str(x).split()))
    df = df[(df['text_len'] >= 10) & (df['text_len'] <= 60)]
    
    # Sample the dataset
    df_sample = df.sample(n=min(total_samples, len(df)), random_state=42)
    
    print("Loading base GLiNER model for zero-shot auto-annotation...")
    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    labels = ["brand", "product", "feature", "issue", "opinion"]
    
    annotated_data = []
    
    print(f"Auto-annotating {len(df_sample)} reviews...")
    for idx, row in df_sample.iterrows():
        text = str(row['review_text'])
        
        tokens_info = tokenize_with_offsets(text)
        token_strings = [t["token"] for t in tokens_info]
        
        entities = model.predict_entities(text, labels, threshold=0.4)
        ner_spans = []
        
        for ent in entities:
            ent_start_char = ent["start"]
            ent_end_char = ent["end"]
            
            start_tok_idx = -1
            end_tok_idx = -1
            
            for i, t in enumerate(tokens_info):
                if t["start"] <= ent_start_char < t["end"] or t["start"] == ent_start_char:
                    if start_tok_idx == -1:
                        start_tok_idx = i
                if t["start"] < ent_end_char <= t["end"] or t["end"] == ent_end_char:
                    end_tok_idx = i
                    
            if start_tok_idx != -1 and end_tok_idx != -1 and start_tok_idx <= end_tok_idx:
                ner_spans.append([start_tok_idx, end_tok_idx, ent["label"]])
                
        # Only keep reviews where the model found at least one entity
        if ner_spans:
            annotated_data.append({
                "tokenized_text": token_strings,
                "ner": ner_spans
            })

    # Shuffle the data before splitting
    random.seed(42)
    random.shuffle(annotated_data)
    
    # Split into 80% Train, 10% Validation, 10% Test
    total = len(annotated_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = annotated_data[:train_end]
    val_data = annotated_data[train_end:val_end]
    test_data = annotated_data[val_end:]
    
    # Save the splits
    splits = {
        "train.json": train_data,
        "validation.json": val_data,
        "test.json": test_data
    }
    
    for filename, data in splits.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {filename}")

    print("\n[!] Zero-shot bootstrapping complete.")
    print("[!] CRITICAL NEXT STEP: Open train.json in VS Code and manually correct any mislabeled spans before uploading to Colab.")

if __name__ == "__main__":
    # Ensure cleaned_amazon_reviews.csv is in the same directory
    auto_annotate_and_split(
        csv_path="cleaned_amazon_reviews.csv", 
        total_samples=1000  # Adjust this number based on how much data you want to fine-tune on
    )