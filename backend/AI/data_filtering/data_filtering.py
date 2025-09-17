import argparse
import json
import os
from typing import List, Dict, Any, Iterator, Iterable

import pandas as pd


def read_input_csv(input_path: str) -> pd.DataFrame:
    """Read CSV and return DataFrame; cleans up common encoding artifacts."""
    # This function reads a CSV file from the given path into a pandas DataFrame.
    # Pandas is a powerful library for data manipulation and analysis.
    df = pd.read_csv(
        input_path,
        header=0,  # The first row of the CSV is the header (column names).
        # Specify the data type for each column to prevent misinterpretation.
        dtype={"user_id": str, "timestamp": str, "sender": str, "message": str},
        encoding="utf-8",  # Use UTF-8 encoding, which is standard for text.
        on_bad_lines="skip",  # If a row has too many columns, skip it.
        quoting=0,  # Disable special quote handling.
        engine="python",  # Use the Python engine for more features like 'on_bad_lines'.
    )
    # Normalize column names to lowercase and remove leading/trailing spaces.
    # This makes accessing columns more predictable (e.g., 'User_ID ' becomes 'user_id').
    df.columns = [c.strip().lower() for c in df.columns]
    # Basic cleanup for the 'message' column.
    if "message" in df.columns:
        df["message"] = (
            df["message"].astype(str)  # Ensure all messages are strings.
            .str.replace("\uFFFD", " ", regex=False)  # Replace unknown characters with a space.
            .str.strip()  # Remove leading/trailing whitespace from messages.
        )
    return df


def filter_user_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the DataFrame to only include messages sent by 'user'."""
    # Check if the required 'sender' column exists. If not, raise an error.
    if "sender" not in df.columns:
        raise ValueError("CSV must include a 'sender' column")
    # Select rows where the 'sender' column (converted to lowercase) is 'user'.
    # .copy() is used to avoid a SettingWithCopyWarning from pandas.
    filtered = df[df["sender"].str.lower() == "user"].copy()
    # Further filter out rows where the 'message' is empty or just whitespace.
    filtered = filtered[filtered["message"].notna() & (filtered["message"].str.strip() != "")]
    return filtered


def load_pipelines(device: int | str = "cpu"):
    """Loads the Hugging Face transformer pipelines for classification."""
    # This function lazy-imports 'transformers' only when needed.
    from transformers import pipeline

    print("Loading classification models... (This may take a while)")

    # A 'pipeline' from the Hugging Face library simplifies using complex models.
    # It handles all the steps: tokenization, model inference, and decoding results.

    # Load a pre-trained model for emotion classification in English.
    # 'j-hartmann/emotion-english-roberta-large' is a model fine-tuned for detecting emotions.
    emotion_classifier = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-roberta-large",
        top_k=None,  # Return scores for all labels, not just the top one.
        device=device,  # Run the model on 'cpu' or a specific GPU (e.g., 0).
    )

    # Load a pre-trained model for zero-shot classification.
    # 'roberta-large-mnli' is a general-purpose model that can classify text
    # against custom labels without being specifically trained on them.
    mental_classifier = pipeline(
        task="zero-shot-classification",
        model="roberta-large-mnli",
        device=device,
    )

    print("Models loaded successfully.")
    return emotion_classifier, mental_classifier


def batched(iterable: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    """A helper function to yield items from an iterable in batches (chunks)."""
    # This is a memory-efficient way to process large amounts of data.
    # Instead of loading everything into memory at once, we process it in small batches.
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch  # 'yield' turns this function into a generator.
            batch = []
    if batch:  # Yield the last, possibly smaller, batch.
        yield batch


def classify_emotions(emotion_pipeline, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
    """Classifies a list of texts for emotion and returns a list of score dictionaries."""
    results: List[Dict[str, float]] = []
    print(f"Classifying emotions for {len(texts)} messages...")
    # Process the texts in batches to avoid overwhelming the model or running out of memory.
    for chunk in batched(texts, batch_size):
        # The pipeline can process a list of texts at once, which is faster than one by one.
        preds = emotion_pipeline(chunk, top_k=None, truncation=True) # type: ignore
        # The model returns a list of lists of dictionaries. We simplify it.
        for pred in preds:
            # Convert the list of {'label': 'anger', 'score': 0.9} dicts
            # into a single dictionary like {'anger': 0.9, 'joy': 0.1, ...}
            label_scores: Dict[str, float] = {p["label"].lower(): float(p["score"]) for p in pred} # type: ignore
            results.append(label_scores)
    return results


def classify_mental(mental_pipeline, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
    """Zero-shot classifies texts for potential mental conditions."""
    # These are the custom labels we want the model to check for in the text.
    candidate_labels = [
        "depression",
        "anxiety",
        "stress",
        "burnout",
        "insomnia",
        "none",  # A neutral or non-relevant category.
    ]
    results: List[Dict[str, float]] = []
    print(f"Classifying mental conditions for {len(texts)} messages...")
    for chunk in batched(texts, batch_size):
        # Run the zero-shot pipeline on a batch of texts.
        outputs = mental_pipeline(
            chunk,
            candidate_labels=candidate_labels,
            multi_label=True,  # Allows a text to be associated with multiple labels.
            hypothesis_template="This text indicates {}.",  # A template to frame the classification task.
            truncation=True,  # Truncate long texts to fit the model's input size.
        )
        # The pipeline's output format can vary, so we normalize it to always be a list.
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            # Create a dictionary mapping each label to its score.
            label_to_score = {lbl.lower(): float(score) for lbl, score in zip(out["labels"], out["scores"])} # type: ignore
            
            # Post-processing logic: If no single label has a high score (e.g., > 0.35),
            # we increase the score for 'none' to make it the likely top choice.
            # This helps reduce false positives for sensitive topics.
            if max(label_to_score.values(), default=0.0) < 0.35:
                label_to_score["none"] = max(label_to_score.get("none", 0.0), 0.75)
            results.append(label_to_score)
    return results


def pick_top_label(score_dict: Dict[str, float]) -> str:
    """Finds the label with the highest score in a dictionary."""
    if not score_dict:
        return "unknown"
    # `max()` with a `key` function finds the item with the maximum value.
    # `kv[1]` refers to the score in the (label, score) item tuple.
    return max(score_dict.items(), key=lambda kv: kv[1])[0]


def init_firestore_client(service_account_json: str | None, project_id: str | None):
    """Initialize Firestore client using optional service account json and project id."""
    try:
        # Lazy-import to avoid making 'google-cloud-firestore' a hard dependency
        # if the user only wants to use CSV files.
        from google.cloud import firestore
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-firestore not installed. Install with: pip install google-cloud-firestore"
        ) from exc

    # If a service account JSON file is provided and exists, use it to authenticate.
    # This is common for running scripts in environments without default credentials.
    if service_account_json and os.path.isfile(service_account_json):
        print(f"Initializing Firestore with service account: {service_account_json}")
        return firestore.Client.from_service_account_json(service_account_json, project=project_id)
    
    # Otherwise, initialize using Application Default Credentials (ADC).
    # This works automatically in Google Cloud environments (like Cloud Run, GCE).
    print("Initializing Firestore with default credentials.")
    return firestore.Client(project=project_id)


def fetch_messages_from_firestore(client, collection_path: str) -> pd.DataFrame:
    """Fetch documents from a Firestore collection and normalize into expected DataFrame."""
    print(f"Fetching messages from Firestore collection: {collection_path}")
    coll_ref = client.collection(collection_path)
    docs = list(coll_ref.stream())  # Get all documents from the collection.
    records: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}  # Convert Firestore document to a Python dictionary.
        # Create a record with expected keys, using .get() for safety if a key is missing.
        records.append(
            {
                "_doc_id": d.id,
                "user_id": str(data.get("user_id", "")),
                "timestamp": str(data.get("timestamp", "")),
                "sender": str(data.get("sender", "")),
                "message": str(data.get("message", "")),
            }
        )
    # Create a pandas DataFrame from the list of records.
    df = pd.DataFrame.from_records(records) # type: ignore
    if df.empty:
        print("No documents found in the collection.")
        return df
    # Ensure all expected columns exist, even if they were missing in Firestore.
    for col in ["user_id", "timestamp", "sender", "message"]:
        if col not in df.columns:
            df[col] = ""
    return df


def write_labels_to_firestore(client, collection_path: str, labeled_df: pd.DataFrame):
    """Write labeled results to a Firestore collection. One doc per message."""
    print(f"Writing {len(labeled_df)} labeled messages to Firestore collection: {collection_path}")
    coll_ref = client.collection(collection_path)
    # Iterate over each row in the DataFrame to create a new Firestore document.
    for _, row in labeled_df.iterrows():
        # Prepare the data payload for Firestore.
        payload = {
            "user_id": row.get("user_id", ""),
            "timestamp": row.get("timestamp", ""),
            "sender": row.get("sender", ""),
            "message": row.get("message", ""),
            "emotion_label": row.get("emotion_label", ""),
            "emotion_scores": json.loads(row.get("emotion_scores", "{}")) if isinstance(row.get("emotion_scores"), str) else row.get("emotion_scores"),
            "mental_label": row.get("mental_label", ""),
            "mental_scores": json.loads(row.get("mental_scores", "{}")) if isinstance(row.get("mental_scores"), str) else row.get("mental_scores"),
        }
        # If we have the original document ID, include it for easy reference.
        if "_doc_id" in labeled_df.columns and pd.notna(row.get("_doc_id")):
            payload["source_doc_id"] = row.get("_doc_id")
        
        # Add a new document to the target collection with the payload.
        coll_ref.add(payload)


def main():
    """The main function that orchestrates the entire script."""
    # `argparse` is the standard Python library for creating command-line interfaces.
    # It allows users to provide arguments like file paths and settings when running the script.
    parser = argparse.ArgumentParser(description="Filter user messages and label emotions and mental conditions using RoBERTa.")
    parser.add_argument("--input", required=False, default=None, help="Path to input CSV. If omitted, reads from Firestore if configured, else uses sample data.")
    parser.add_argument("--output", required=False, default="labeled_output.csv", help="Path to write labeled CSV.")
    parser.add_argument("--device", required=False, default="cpu", help="Device to run models on ('cpu' or GPU index like '0').")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for emotion model.")
    parser.add_argument("--mental_batch", type=int, default=8, help="Batch size for mental model.")
    # Firestore-specific command-line options
    parser.add_argument("--firestore_input_collection", required=False, default=None, help="Firestore collection path to read messages from.")
    parser.add_argument("--firestore_output_collection", required=False, default=None, help="Firestore collection path to write labeled results to.")
    parser.add_argument("--firestore_project", required=False, default=None, help="GCP project id (optional if service account provides it).")
    parser.add_argument("--firestore_service_account", required=False, default=None, help="Path to service account JSON file.")
    args = parser.parse_args()

    # --- Step 1: Load Input Data ---
    # The script supports three ways to get input data, in order of priority:
    # 1. A CSV file specified with `--input`.
    # 2. A Firestore collection specified with `--firestore_input_collection`.
    # 3. A hardcoded sample data if no other input is given.

    if args.input is None:
        # No --input CSV provided, try Firestore.
        if args.firestore_input_collection:
            try:
                fs_client = init_firestore_client(args.firestore_service_account, args.firestore_project)
                df = fetch_messages_from_firestore(fs_client, args.firestore_input_collection)
            except Exception as exc:
                print(f"Failed to read from Firestore: {exc}")
                df = pd.DataFrame()  # Create an empty DataFrame on failure.
        else:
            # No CSV or Firestore input, so fall back to the sample data.
            print("No input file or Firestore collection specified. Using sample data.")
            from io import StringIO

            # StringIO allows treating a string as a file, which is useful for `pd.read_csv`.
            sample = StringIO(
                """user_id,timestamp,sender,message
U001,05-09-2025 09:15,user,"Hey! I'm so excited today, I finally got promoted!"
U001,05-09-2025 09:15,bot,"That's amazing, congratulations!"
U001,05-09-2025 09:15,user,Thank you! I'm feeling so happy right now.
U002,05-09-2025 10:20,user,"Work is okay, just the usual stuff."
U002,05-09-2025 10:20,bot,Anything stressful happening?
U002,05-09-2025 10:20,user,"Not really, just a normal day."
U003,05-09-2025 11:45,user,I'm so tired� haven't slept properly in three days.
U003,05-09-2025 11:45,bot,What's wrong? Are you okay?
U003,05-09-2025 11:45,user,Too much work pressure and constant deadlines. Feeling anxious.
U004,05-09-2025 12:30,user,I had a fight with my parents today.
U004,05-09-2025 12:30,bot,"Oh no, what happened?"
U004,05-09-2025 12:30,user,They don't understand me� feeling frustrated and alone.
U005,05-09-2025 14:10,user,"Just watched a comedy movie, laughing so hard ??"
U005,05-09-2025 14:10,bot,"Haha, I need to watch it too!"
U005,05-09-2025 14:10,user,"You should, it's super funny!"
"""
            )
            df = read_input_csv(sample)
    else:
        # An --input CSV was provided.
        print(f"Reading from input file: {args.input}")
        df = read_input_csv(args.input)

    # --- Step 2: Filter and Prepare Data ---
    users_df = filter_user_messages(df)
    if users_df.empty:
        print("No user messages found to classify. Exiting.")
        return

    # --- Step 3: Load AI Models ---
    try:
        # Determine if the device is a GPU (integer) or CPU (string).
        device = int(args.device) if str(args.device).isdigit() else args.device
        emotion_pipe, mental_pipe = load_pipelines(device)
    except Exception as exc:
        print("Failed to load models. Ensure 'transformers', 'torch', and 'sentencepiece' are installed.")
        print(f"Error: {exc}")
        print("Try installing with: pip install -U transformers torch sentencepiece pandas")
        return

    # Get the list of messages to be classified.
    texts: List[str] = users_df["message"].astype(str).tolist()

    # --- Step 4: Run Classifications ---
    # Classify emotions for all user messages.
    emotion_scores = classify_emotions(emotion_pipe, texts, batch_size=args.batch)
    emotion_labels = [pick_top_label(s) for s in emotion_scores]

    # Classify mental conditions for all user messages.
    mental_scores = classify_mental(mental_pipe, texts, batch_size=args.mental_batch)
    mental_labels = [pick_top_label(s) for s in mental_scores]

    # --- Step 5: Combine Results ---
    # Create a new DataFrame to store the results.
    out_df = users_df.copy()
    out_df["emotion_label"] = emotion_labels  # Add the top emotion label as a new column.
    # Add the full dictionary of emotion scores, converted to a JSON string for CSV compatibility.
    out_df["emotion_scores"] = [json.dumps(s, ensure_ascii=False) for s in emotion_scores] 
    out_df["mental_label"] = mental_labels  # Add the top mental condition label.
    out_df["mental_scores"] = [json.dumps(s, ensure_ascii=False) for s in mental_scores]

    # --- Step 6: Write Output ---
    # Always write the results to a CSV file.
    out_path = os.path.abspath(args.output)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote labeled results to: {out_path}")

    # Optionally, if a Firestore output collection was specified, write the results there too.
    if args.firestore_output_collection:
        try:
            # Re-initialize client in case it wasn't for input.
            fs_client = init_firestore_client(args.firestore_service_account, args.firestore_project)
            write_labels_to_firestore(fs_client, args.firestore_output_collection, out_df)
            print(f"Successfully wrote labeled results to Firestore collection: {args.firestore_output_collection}")
        except Exception as exc:
            print(f"Failed to write to Firestore: {exc}")


# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly (e.g., `python data_filtering.py`),
# not when it's imported as a module into another script.
if __name__ == "__main__":
    main()
