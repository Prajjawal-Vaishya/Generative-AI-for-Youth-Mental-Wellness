import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd


def read_input_csv(input_path: str) -> pd.DataFrame:
    """Read CSV and return DataFrame; cleans up common encoding artifacts."""
    df = pd.read_csv(
        input_path,
        header=0,
        dtype={"user_id": str, "timestamp": str, "sender": str, "message": str},
        encoding="utf-8",
        on_bad_lines="skip",
        quoting=0,
        engine="python",
    )
    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    # Basic cleanup
    if "message" in df.columns:
        df["message"] = (
            df["message"].astype(str).str.replace("\uFFFD", " ", regex=False).str.strip()
        )
    return df


def filter_user_messages(df: pd.DataFrame) -> pd.DataFrame:
    if "sender" not in df.columns:
        raise ValueError("CSV must include a 'sender' column")
    filtered = df[df["sender"].str.lower() == "user"].copy()
    filtered = filtered[filtered["message"].notna() & (filtered["message"].str.strip() != "")]
    return filtered


def load_pipelines(device: int | str = "cpu"):
    from transformers import pipeline

    # Emotion model (English)
    emotion_classifier = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-roberta-large",
        top_k=None,
        device=device,
    )

    # Zero-shot for mental condition detection
    mental_classifier = pipeline(
        task="zero-shot-classification",
        model="roberta-large-mnli",
        device=device,
    )

    return emotion_classifier, mental_classifier


def batched(iterable: List[str], batch_size: int) -> List[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def classify_emotions(emotion_pipeline, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
    """Return list of label->score dicts for each text."""
    results: List[Dict[str, float]] = []
    for chunk in batched(texts, batch_size):
        preds = emotion_pipeline(chunk, top_k=None, truncation=True)
        for pred in preds:
            label_scores: Dict[str, float] = {p["label"].lower(): float(p["score"]) for p in pred}
            results.append(label_scores)
    return results


def classify_mental(mental_pipeline, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
    """Zero-shot classify potential mental conditions and return label->score dict per text."""
    candidate_labels = [
        "depression",
        "anxiety",
        "stress",
        "burnout",
        "insomnia",
        "none",
    ]
    results: List[Dict[str, float]] = []
    for chunk in batched(texts, batch_size):
        outputs = mental_pipeline(
            chunk,
            candidate_labels=candidate_labels,
            multi_label=True,
            hypothesis_template="This text indicates {}.",
            truncation=True,
        )
        # Normalize outputs to dicts
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            label_to_score = {lbl.lower(): float(score) for lbl, score in zip(out["labels"], out["scores"])}
            # Post-process: if no label has meaningful score, prefer 'none'
            if max(label_to_score.values(), default=0.0) < 0.35:
                label_to_score["none"] = max(label_to_score.get("none", 0.0), 0.75)
            results.append(label_to_score)
    return results


def pick_top_label(score_dict: Dict[str, float]) -> str:
    if not score_dict:
        return "unknown"
    return max(score_dict.items(), key=lambda kv: kv[1])[0]


def init_firestore_client(service_account_json: str | None, project_id: str | None):
    """Initialize Firestore client using optional service account json and project id."""
    try:
        from google.cloud import firestore
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-firestore not installed. Install with: pip install google-cloud-firestore"
        ) from exc

    if service_account_json and os.path.isfile(service_account_json):
        return firestore.Client.from_service_account_json(service_account_json, project=project_id)
    return firestore.Client(project=project_id)


def fetch_messages_from_firestore(client, collection_path: str) -> pd.DataFrame:
    """Fetch documents from a Firestore collection and normalize into expected DataFrame."""
    coll_ref = client.collection(collection_path)
    docs = list(coll_ref.stream())
    records: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        records.append(
            {
                "_doc_id": d.id,
                "user_id": str(data.get("user_id", "")),
                "timestamp": str(data.get("timestamp", "")),
                "sender": str(data.get("sender", "")),
                "message": str(data.get("message", "")),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    # Ensure expected columns exist
    for col in ["user_id", "timestamp", "sender", "message"]:
        if col not in df.columns:
            df[col] = ""
    return df


def write_labels_to_firestore(client, collection_path: str, labeled_df: pd.DataFrame):
    """Write labeled results to a Firestore collection. One doc per message."""
    coll_ref = client.collection(collection_path)
    for _, row in labeled_df.iterrows():
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
        # If we have original doc id, include it for traceability
        if "_doc_id" in labeled_df.columns and pd.notna(row.get("_doc_id")):
            payload["source_doc_id"] = row.get("_doc_id")
        coll_ref.add(payload)


def main():
    parser = argparse.ArgumentParser(description="Filter user messages and label emotions and mental conditions using RoBERTa.")
    parser.add_argument("--input", required=False, default=None, help="Path to input CSV. If omitted, reads from Firestore if configured, else uses sample data.")
    parser.add_argument("--output", required=False, default="labeled_output.csv", help="Path to write labeled CSV.")
    parser.add_argument("--device", required=False, default="cpu", help="Device index or 'cpu'. Example: 0 for first GPU.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for emotion model.")
    parser.add_argument("--mental_batch", type=int, default=8, help="Batch size for mental model.")
    # Firestore options
    parser.add_argument("--firestore_input_collection", required=False, default=None, help="Firestore collection path to read messages from.")
    parser.add_argument("--firestore_output_collection", required=False, default=None, help="Firestore collection path to write labeled results to.")
    parser.add_argument("--firestore_project", required=False, default=None, help="GCP project id (optional if service account provides it).")
    parser.add_argument("--firestore_service_account", required=False, default=None, help="Path to service account JSON file.")
    args = parser.parse_args()

    if args.input is None:
        # Try Firestore if configured
        if args.firestore_input_collection:
            try:
                fs_client = init_firestore_client(args.firestore_service_account, args.firestore_project)
                df = fetch_messages_from_firestore(fs_client, args.firestore_input_collection)
            except Exception as exc:
                print(f"Failed to read from Firestore: {exc}")
                df = pd.DataFrame()
        else:
            # Fallback to the sample provided by user
            from io import StringIO

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
        df = read_input_csv(args.input)

    users_df = filter_user_messages(df)
    if users_df.empty:
        print("No user messages found to classify.")
        return

    # Load models
    try:
        device = int(args.device) if str(args.device).isdigit() else args.device
        emotion_pipe, mental_pipe = load_pipelines(device)
    except Exception as exc:
        print("Failed to load models. Ensure transformers, torch, and sentencepiece are installed.")
        print(f"Error: {exc}")
        print("Try installing: pip install -U transformers torch sentencepiece pandas")
        return

    texts: List[str] = users_df["message"].astype(str).tolist()

    # Classify emotions
    emotion_scores = classify_emotions(emotion_pipe, texts, batch_size=args.batch)
    emotion_labels = [pick_top_label(s) for s in emotion_scores]

    # Classify mental conditions
    mental_scores = classify_mental(mental_pipe, texts, batch_size=args.mental_batch)
    mental_labels = [pick_top_label(s) for s in mental_scores]

    # Build output DataFrame
    out_df = users_df.copy()
    out_df["emotion_label"] = emotion_labels
    out_df["emotion_scores"] = [json.dumps(s, ensure_ascii=False) for s in emotion_scores]
    out_df["mental_label"] = mental_labels
    out_df["mental_scores"] = [json.dumps(s, ensure_ascii=False) for s in mental_scores]

    # Write CSV always
    out_path = os.path.abspath(args.output)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote labeled results to: {out_path}")

    # Optionally write to Firestore
    if args.firestore_output_collection:
        try:
            fs_client = init_firestore_client(args.firestore_service_account, args.firestore_project)
            write_labels_to_firestore(fs_client, args.firestore_output_collection, out_df)
            print(f"Wrote labeled results to Firestore collection: {args.firestore_output_collection}")
        except Exception as exc:
            print(f"Failed to write to Firestore: {exc}")


if __name__ == "__main__":
    main()

