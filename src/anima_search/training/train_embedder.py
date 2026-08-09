from __future__ import annotations

import json
import random
from pathlib import Path


def train_embedder(base_model: str | Path, pairs_path: Path, output_dir: Path,
                   epochs: int = 3, batch_size: int = 16, seed: int = 20260802) -> None:
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses

    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(seed)
    rng.shuffle(rows)
    split_at = max(1, int(len(rows) * 0.9))
    train_rows, validation_rows = rows[:split_at], rows[split_at:]
    if not validation_rows:
        validation_rows = rows[-1:]
    examples = [InputExample(texts=[row["query"], row["positive"], row["negative"]]) for row in train_rows]
    model = SentenceTransformer(str(base_model))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model=model)
    queries = {f"q{index}": row["query"] for index, row in enumerate(validation_rows)}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for index, row in enumerate(validation_rows):
        positive_id, negative_id = f"p{index}", f"n{index}"
        corpus[positive_id], corpus[negative_id] = row["positive"], row["negative"]
        relevant_docs[f"q{index}"] = {positive_id}
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        ndcg_at_k=[10], name="train-internal-validation",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=max(1, len(loader) // 10), evaluator=evaluator,
              evaluation_steps=max(1, len(loader)), save_best_model=True,
              output_path=str(output_dir), show_progress_bar=True, use_amp=True)
    (output_dir / "training_metadata.json").write_text(json.dumps(
        {"base_model": str(base_model), "epochs": epochs, "batch_size": batch_size,
         "seed": seed, "pairs": len(rows), "train_pairs": len(train_rows),
         "validation_pairs": len(validation_rows), "loss": "TripletLoss",
         "selection_metric": "nDCG@10"}, ensure_ascii=False, indent=2), encoding="utf-8")
