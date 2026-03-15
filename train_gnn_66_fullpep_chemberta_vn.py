import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader


TRAIN_PT = "/content/drive/MyDrive/master_thesis/sampled_data_5000/GNN/peptide_graphs_train.pt"
VAL_PT = "/content/drive/MyDrive/master_thesis/sampled_data_5000/GNN/peptide_graphs_val.pt"
TEST_PT = "/content/drive/MyDrive/master_thesis/sampled_data_5000/GNN/peptide_graphs_test.pt"
TRAIN_EMBEDDING_PT = "/content/drive/MyDrive/5000_gnn/data/chemberta/chemberta_split_2hop/peptide_embeddings_train.pt"
VAL_EMBEDDING_PT = "/content/drive/MyDrive/5000_gnn/data/chemberta/chemberta_split_2hop/peptide_embeddings_val.pt"
TEST_EMBEDDING_PT = "/content/drive/MyDrive/5000_gnn/data/chemberta/chemberta_split_2hop/peptide_embeddings_test.pt"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3-layer VN GNN fused with whole-peptide ChemBERTa embeddings."
    )
    parser.add_argument("--exp_prefix", type=str, default="66_fullpep_chemberta_gnn_3layer")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "add"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Run all epochs without early stopping.",
    )
    parser.add_argument("--train_pt", type=str, default=TRAIN_PT)
    parser.add_argument("--val_pt", type=str, default=VAL_PT)
    parser.add_argument("--test_pt", type=str, default=TEST_PT)
    parser.add_argument(
        "--train_embedding_pt",
        type=str,
        default=TRAIN_EMBEDDING_PT,
        help="Train split whole-peptide embedding PT file.",
    )
    parser.add_argument(
        "--val_embedding_pt",
        type=str,
        default=VAL_EMBEDDING_PT,
        help="Validation split whole-peptide embedding PT file.",
    )
    parser.add_argument(
        "--test_embedding_pt",
        type=str,
        default=TEST_EMBEDDING_PT,
        help="Test split whole-peptide embedding PT file.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional external directory that contains model_66_fullpep_chemberta_vn.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "outputs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--monitor",
        default="val_pr_auc",
        choices=["val_dual_score", "val_pr_auc", "val_auc", "val_loss"],
        help="Metric used by checkpointing and early stopping.",
    )
    parser.add_argument(
        "--threshold_metric",
        default="dual",
        choices=["dual", "f1", "mcc"],
        help="Validation metric used to select the final classification threshold.",
    )
    return parser.parse_args()


def add_model_root_to_path(model_root: Path):
    model_root = model_root.expanduser().resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"model_root does not exist: {model_root}")
    sys.path.insert(0, str(model_root))


def import_model():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from model_66_fullpep_chemberta_vn import GINEVirtualNodeChemBERTaClassifier

    return GINEVirtualNodeChemBERTaClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        print(f"[WARN] deterministic_algorithms not fully supported: {exc}")
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders_with_seed(train_dataset, val_dataset, test_dataset, batch_size, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen,
    )
    return train_loader, val_loader, test_loader


def compute_pos_weight(train_loader, device):
    y_all = []
    for data in train_loader:
        y_all.append(data.y.view(-1))
    y_all = torch.cat(y_all, dim=0)
    pos = y_all.sum().item()
    neg = len(y_all) - pos
    if pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, device=device)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_parameters_txt(path: Path, args):
    lines = []
    for key, value in sorted(vars(args).items()):
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        lines.append(f"{key}={value}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def build_output_dirs(output_dir: Path):
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "thresholds": output_dir / "thresholds",
        "logs": output_dir / "logs",
        "plots": output_dir / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_peptide_embedding_dict(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = payload["embeddings"].float()
    aa_seq_list = payload.get("aa_seq", [])
    smiles_list = payload.get("smiles", [])
    row_id_list = payload.get("row_id", [])

    aa_seq_to_emb = {}
    smiles_to_emb = {}
    row_id_to_emb = {}

    for idx, emb in enumerate(embeddings):
        emb_vec = emb.unsqueeze(0).clone()
        if idx < len(aa_seq_list):
            key = str(aa_seq_list[idx]).strip()
            if key and key not in aa_seq_to_emb:
                aa_seq_to_emb[key] = emb_vec
        if idx < len(smiles_list):
            key = str(smiles_list[idx]).strip()
            if key and key not in smiles_to_emb:
                smiles_to_emb[key] = emb_vec
        if idx < len(row_id_list):
            row_id_to_emb[int(row_id_list[idx])] = emb_vec

    return {
        "dim": int(embeddings.shape[1]),
        "aa_seq": aa_seq_to_emb,
        "smiles": smiles_to_emb,
        "row_id": row_id_to_emb,
        "model_name": payload.get("model_name"),
        "source_file": payload.get("source_file"),
        "split": payload.get("split"),
    }


def resolve_data_key(data):
    for attr in ["aa_seq", "sequence", "smiles", "row_id"]:
        if hasattr(data, attr):
            value = getattr(data, attr)
            if value is not None:
                if attr == "row_id":
                    return attr, int(value)
                return attr, str(value).strip()
    raise ValueError(
        "Could not find a matching identifier on graph data. "
        "Expected one of: aa_seq, sequence, smiles, row_id."
    )


def attach_embeddings_to_data_list(data_list, embedding_lookup, split_name: str):
    attached = []
    missing = []

    for idx, data in enumerate(data_list):
        key_name, key_value = resolve_data_key(data)
        if key_name == "sequence":
            emb = embedding_lookup["aa_seq"].get(key_value)
        else:
            emb = embedding_lookup[key_name].get(key_value)

        if emb is None:
            missing.append((idx, key_name, key_value))
            continue

        new_data = copy.deepcopy(data)
        new_data.peptide_embedding = emb.clone()
        attached.append(new_data)

    if missing:
        example = missing[0]
        raise ValueError(
            f"Failed to match whole-peptide embeddings for {split_name} graph samples. "
            f"First missing sample: index={example[0]}, key={example[1]}, value={example[2]!r}. "
            "Make sure the graph PT data contains aa_seq/sequence/smiles/row_id "
            "that matches the embedding file."
        )

    return attached


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        y = data.y.view(-1).float()
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n_graphs += data.num_graphs
    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n_graphs = 0
    all_probs = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        y = data.y.view(-1).float()
        loss = loss_fn(logits, y)
        total_loss += loss.item() * data.num_graphs
        n_graphs += data.num_graphs
        probs = torch.sigmoid(logits).detach().cpu()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / max(n_graphs, 1)
    probs = torch.cat(all_probs).numpy() if all_probs else np.array([])
    labels = torch.cat(all_labels).numpy() if all_labels else np.array([])

    roc_auc = None
    pr_auc = None
    if len(labels) > 0 and len(np.unique(labels)) >= 2:
        roc_auc = float(roc_auc_score(labels, probs))
        pr_auc = float(average_precision_score(labels, probs))

    preds = (probs >= threshold).astype(int) if len(probs) > 0 else np.array([])
    acc = float(accuracy_score(labels, preds)) if len(labels) > 0 else 0.0
    prec = float(precision_score(labels, preds, zero_division=0)) if len(labels) > 0 else 0.0
    recall = float(recall_score(labels, preds, zero_division=0)) if len(labels) > 0 else 0.0
    f1 = float(f1_score(labels, preds, zero_division=0)) if len(labels) > 0 else 0.0
    mcc = float(matthews_corrcoef(labels, preds)) if len(labels) > 0 else 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1]) if len(labels) > 0 else None

    return {
        "loss": float(avg_loss),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm,
        "probs": probs,
        "labels": labels,
        "preds": preds,
    }


def score_thresholds(y_true, y_prob):
    rows = []
    thresholds = np.linspace(0.0, 1.0, 1001)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        mcc_norm = float((mcc + 1.0) / 2.0)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1,
                "mcc": mcc,
                "mcc_norm": mcc_norm,
                "dual_score": float((f1 + mcc_norm) / 2.0),
                "balanced_score": float(min(f1, mcc_norm)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(y_true, y_prob, metric_name: str):
    threshold_df = score_thresholds(y_true, y_prob)
    if metric_name == "dual":
        best_row = threshold_df.sort_values(
            by=["dual_score", "balanced_score", "f1", "mcc", "threshold"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
    else:
        best_row = threshold_df.sort_values(
            by=[metric_name, "threshold"], ascending=[False, True]
        ).iloc[0]
    return float(best_row["threshold"]), threshold_df, best_row.to_dict()


def get_monitor_value(monitor_name: str, val_metrics: dict, threshold_df: pd.DataFrame):
    if monitor_name == "val_dual_score":
        return float(threshold_df["dual_score"].max())
    if monitor_name == "val_pr_auc":
        return float(val_metrics["pr_auc"])
    if monitor_name == "val_auc":
        return float(val_metrics["roc_auc"])
    if monitor_name == "val_loss":
        return float(val_metrics["loss"])
    raise ValueError(f"Unsupported monitor: {monitor_name}")


def evaluate_split(y_true, y_prob, loss, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y_true)),
        "loss": float(loss),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
    }


def save_prediction_table(path: Path, probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    df = pd.DataFrame(
        {
            "label": labels.astype(int),
            "pred_prob": probs.astype(float),
            "pred_label": preds.astype(int),
        }
    )
    df.to_csv(path, sep="\t", index=False)


def train_with_canya_strategy(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    loss_fn,
    epochs,
    patience,
    disable_early_stopping,
    monitor,
    threshold_metric,
    output_dirs,
):
    model = model.to(device)
    mode = "min" if monitor == "val_loss" else "max"
    best_value = np.inf if mode == "min" else -np.inf
    best_epoch = -1
    best_metrics = None
    best_model_path = output_dirs["checkpoints"] / "best_model.pt"
    epoch_records = []
    history_rows = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=0.5,
        patience=max(2, patience // 2),
        threshold=1e-4,
        min_lr=1e-6,
    )

    wait = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_raw = evaluate(model, val_loader, device, loss_fn, threshold=0.5)
        selected_threshold, threshold_df, selected_row = select_threshold(
            val_raw["labels"], val_raw["probs"], metric_name=threshold_metric
        )
        monitor_value = get_monitor_value(monitor, val_raw, threshold_df)
        scheduler.step(monitor_value)
        lr = float(optimizer.param_groups[0]["lr"])

        epoch_records.append(
            {
                "epoch": int(epoch),
                "threshold": float(selected_threshold),
                "f1": float(selected_row["f1"]),
                "mcc": float(selected_row["mcc"]),
                "dual_score": float(selected_row["dual_score"]),
                "balanced_score": float(selected_row["balanced_score"]),
                "precision": float(selected_row["precision"]),
                "recall": float(selected_row["recall"]),
                "accuracy": float(selected_row["accuracy"]),
                "val_loss": float(val_raw["loss"]),
                "val_auc": None if val_raw["roc_auc"] is None else float(val_raw["roc_auc"]),
                "val_pr_auc": None if val_raw["pr_auc"] is None else float(val_raw["pr_auc"]),
                "monitor_name": monitor,
                "monitor_value": float(monitor_value),
                "learning_rate": lr,
            }
        )
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_raw["loss"]),
                "val_auc": None if val_raw["roc_auc"] is None else float(val_raw["roc_auc"]),
                "val_pr_auc": None if val_raw["pr_auc"] is None else float(val_raw["pr_auc"]),
                "val_dual_score": float(threshold_df["dual_score"].max()),
                "selected_threshold": float(selected_threshold),
                "selected_f1": float(selected_row["f1"]),
                "selected_mcc": float(selected_row["mcc"]),
                "selected_precision": float(selected_row["precision"]),
                "selected_recall": float(selected_row["recall"]),
                "learning_rate": lr,
                "monitor_name": monitor,
                "monitor_value": float(monitor_value),
            }
        )

        improved = monitor_value < best_value if mode == "min" else monitor_value > best_value
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_raw['loss']:.4f} | val_auc={val_raw['roc_auc'] if val_raw['roc_auc'] is not None else float('nan'):.4f} | "
            f"val_pr_auc={val_raw['pr_auc'] if val_raw['pr_auc'] is not None else float('nan'):.4f} | "
            f"threshold={selected_threshold:.3f} | f1={selected_row['f1']:.4f} | "
            f"mcc={selected_row['mcc']:.4f} | dual_score={selected_row['dual_score']:.4f} | "
            f"{monitor}={monitor_value:.4f}"
        )

        if improved:
            best_value = float(monitor_value)
            best_epoch = int(epoch)
            best_metrics = {
                "f1": float(selected_row["f1"]),
                "mcc": float(selected_row["mcc"]),
                "dual_score": float(selected_row["dual_score"]),
                "balanced_score": float(selected_row["balanced_score"]),
                "precision": float(selected_row["precision"]),
                "recall": float(selected_row["recall"]),
                "accuracy": float(selected_row["accuracy"]),
                "val_loss": float(val_raw["loss"]),
                "val_auc": None if val_raw["roc_auc"] is None else float(val_raw["roc_auc"]),
                "val_pr_auc": None if val_raw["pr_auc"] is None else float(val_raw["pr_auc"]),
            }
            torch.save(model.state_dict(), best_model_path)
            wait = 0
            print(f"Saved best model at epoch {epoch} using {monitor}={monitor_value:.4f}")
        else:
            wait += 1
            if not disable_early_stopping and wait >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    pd.DataFrame(history_rows).to_csv(output_dirs["logs"] / "history.tsv", sep="\t", index=False)
    pd.DataFrame(epoch_records).to_csv(
        output_dirs["thresholds"] / "epoch_level_validation_metrics.tsv",
        sep="\t",
        index=False,
    )
    return {
        "best_model_path": best_model_path,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "best_metrics": best_metrics,
        "epochs_completed": len(history_rows),
    }


def run_single_seed(args, model_cls, train_list, val_list, test_list, seed: int, run_dir: Path, peptide_emb_dim: int):
    output_dirs = build_output_dirs(run_dir)
    set_seed(seed)

    train_loader, val_loader, test_loader = build_loaders_with_seed(
        train_list, val_list, test_list, batch_size=args.batch_size, seed=seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(
        in_dim=train_list[0].x.size(1),
        peptide_emb_dim=peptide_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        edge_attr_dim=train_list[0].edge_attr.size(1),
    )
    pos_weight = compute_pos_weight(train_loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_summary = train_with_canya_strategy(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
        monitor=args.monitor,
        threshold_metric=args.threshold_metric,
        output_dirs=output_dirs,
    )

    model.load_state_dict(torch.load(train_summary["best_model_path"], map_location=device))
    model = model.to(device)

    val_raw = evaluate(model, val_loader, device, loss_fn, threshold=0.5)
    test_raw = evaluate(model, test_loader, device, loss_fn, threshold=0.5)
    selected_threshold, threshold_df, _ = select_threshold(
        val_raw["labels"], val_raw["probs"], metric_name=args.threshold_metric
    )
    threshold_df.to_csv(
        output_dirs["thresholds"] / "validation_threshold_scan.tsv",
        sep="\t",
        index=False,
    )

    val_metrics = evaluate_split(val_raw["labels"], val_raw["probs"], val_raw["loss"], selected_threshold)
    test_metrics = evaluate_split(test_raw["labels"], test_raw["probs"], test_raw["loss"], selected_threshold)

    save_prediction_table(
        output_dirs["predictions"] / "validation_predictions.tsv",
        val_raw["probs"],
        val_raw["labels"],
        selected_threshold,
    )
    save_prediction_table(
        output_dirs["predictions"] / "test_predictions.tsv",
        test_raw["probs"],
        test_raw["labels"],
        selected_threshold,
    )

    training_summary = {
        "seed": int(seed),
        "monitor": args.monitor,
        "threshold_metric": args.threshold_metric,
        "selected_threshold": float(selected_threshold),
        "epochs_completed": int(train_summary["epochs_completed"]),
        "best_weights": str(train_summary["best_model_path"]),
        "best_epoch": int(train_summary["best_epoch"]),
        "best_monitor_value": float(train_summary["best_value"]),
        "best_epoch_metrics": train_summary["best_metrics"],
        "train_samples": int(len(train_list)),
        "val_samples": int(len(val_list)),
        "test_samples": int(len(test_list)),
        "peptide_embedding_dim": int(peptide_emb_dim),
    }

    save_json(output_dirs["metrics"] / "training_summary.json", training_summary)
    save_json(output_dirs["metrics"] / "validation_metrics.json", val_metrics)
    save_json(output_dirs["metrics"] / "test_metrics.json", test_metrics)
    return {
        "seed": int(seed),
        "run_dir": str(run_dir),
        "training_summary": training_summary,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def summarize_multi_seed(results, output_dir: Path):
    rows = []
    for result in results:
        rows.append(
            {
                "seed": int(result["seed"]),
                "selected_threshold": float(result["training_summary"]["selected_threshold"]),
                "best_epoch": int(result["training_summary"]["best_epoch"]),
                "best_monitor_value": float(result["training_summary"]["best_monitor_value"]),
                "val_accuracy": float(result["validation_metrics"]["accuracy"]),
                "val_f1": float(result["validation_metrics"]["f1"]),
                "val_mcc": float(result["validation_metrics"]["mcc"]),
                "val_precision": float(result["validation_metrics"]["precision"]),
                "val_recall": float(result["validation_metrics"]["recall"]),
                "val_roc_auc": float(result["validation_metrics"]["roc_auc"]),
                "val_pr_auc": float(result["validation_metrics"]["pr_auc"]),
                "test_accuracy": float(result["test_metrics"]["accuracy"]),
                "test_f1": float(result["test_metrics"]["f1"]),
                "test_mcc": float(result["test_metrics"]["mcc"]),
                "test_precision": float(result["test_metrics"]["precision"]),
                "test_recall": float(result["test_metrics"]["recall"]),
                "test_roc_auc": float(result["test_metrics"]["roc_auc"]),
                "test_pr_auc": float(result["test_metrics"]["pr_auc"]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary_df.to_csv(output_dir / "multi_seed_metrics.tsv", sep="\t", index=False)
    metric_cols = [col for col in summary_df.columns if col != "seed"]
    aggregate = {
        "n_seeds": int(len(summary_df)),
        "seeds": summary_df["seed"].astype(int).tolist(),
        "mean": {col: float(summary_df[col].mean()) for col in metric_cols},
        "std": {
            col: float(summary_df[col].std(ddof=1)) if len(summary_df) > 1 else 0.0
            for col in metric_cols
        },
    }
    save_json(output_dir / "multi_seed_summary.json", aggregate)


def main():
    args = parse_args()
    if args.model_root:
        add_model_root_to_path(Path(args.model_root))
    model_cls = import_model()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters_txt(output_dir / "parameters.txt", args)

    train_embedding_lookup = load_peptide_embedding_dict(
        Path(args.train_embedding_pt).expanduser()
    )
    val_embedding_lookup = load_peptide_embedding_dict(
        Path(args.val_embedding_pt).expanduser()
    )
    test_embedding_lookup = load_peptide_embedding_dict(
        Path(args.test_embedding_pt).expanduser()
    )
    print(
        "Loaded train peptide embeddings: "
        f"dim={train_embedding_lookup['dim']} "
        f"model={train_embedding_lookup['model_name']} "
        f"source={train_embedding_lookup['source_file']}"
    )
    print(
        "Loaded val peptide embeddings: "
        f"dim={val_embedding_lookup['dim']} "
        f"model={val_embedding_lookup['model_name']} "
        f"source={val_embedding_lookup['source_file']}"
    )
    print(
        "Loaded test peptide embeddings: "
        f"dim={test_embedding_lookup['dim']} "
        f"model={test_embedding_lookup['model_name']} "
        f"source={test_embedding_lookup['source_file']}"
    )

    dims = {
        train_embedding_lookup["dim"],
        val_embedding_lookup["dim"],
        test_embedding_lookup["dim"],
    }
    if len(dims) != 1:
        raise ValueError(f"Embedding dimension mismatch across splits: {sorted(dims)}")

    train_list = torch.load(args.train_pt, map_location="cpu", weights_only=False)["data_list"]
    val_list = torch.load(args.val_pt, map_location="cpu", weights_only=False)["data_list"]
    test_list = torch.load(args.test_pt, map_location="cpu", weights_only=False)["data_list"]

    train_list = attach_embeddings_to_data_list(train_list, train_embedding_lookup, "train")
    val_list = attach_embeddings_to_data_list(val_list, val_embedding_lookup, "val")
    test_list = attach_embeddings_to_data_list(test_list, test_embedding_lookup, "test")

    total = len(train_list) + len(val_list) + len(test_list)
    print(
        "Dataset split sizes: "
        f"train={len(train_list)} ({len(train_list) / total:.2%}), "
        f"val={len(val_list)} ({len(val_list) / total:.2%}), "
        f"test={len(test_list)} ({len(test_list) / total:.2%}), "
        f"total={total}"
    )

    results = []
    for seed in args.seeds:
        print(f"\n=== Run seed={seed} ===")
        run_dir = output_dir / f"seed_{seed}"
        result = run_single_seed(
            args,
            model_cls,
            train_list,
            val_list,
            test_list,
            seed,
            run_dir,
            peptide_emb_dim=train_embedding_lookup["dim"],
        )
        results.append(result)

    if len(results) > 1:
        summarize_multi_seed(results, output_dir)
        print(f"\nSaved multi-seed summary to {output_dir}")


if __name__ == "__main__":
    main()
