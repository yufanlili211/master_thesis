import argparse
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
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GNN_66 with CANYA-style checkpointing, early stopping, and threshold selection."
    )
    parser.add_argument("--exp_prefix", type=str, default="66")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "add"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
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
        "--model_root",
        type=str,
        default=None,
        help="Optional external directory that contains model_66.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "66_outputs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--monitor",
        default="val_f1",
        choices=["val_f1", "val_dual_score", "val_pr_auc", "val_auc", "val_loss"],
        help="Metric used by checkpointing and early stopping.",
    )
    parser.add_argument(
        "--threshold_metric",
        default="f1",
        choices=["dual", "f1", "mcc"],
        help="Validation metric used to select the final classification threshold.",
    )
    parser.add_argument(
        "--pretrained_encoder_path",
        type=str,
        default=None,
        help=(
            "Optional path to a proxy-task pretrained encoder checkpoint. "
            "If provided, compatible encoder weights will be loaded before finetuning."
        ),
    )
    parser.add_argument(
        "--skip_pretrained_edge_encoder",
        action="store_true",
        help="Skip loading edge_encoder.* from pretrained weights.",
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
    from model_66 import GINEVirtualNodeClassifier

    return GINEVirtualNodeClassifier


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


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ["state_dict", "model_state_dict", "encoder_state_dict"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint format for pretrained encoder.")


def load_pretrained_encoder_weights(model, checkpoint_path: Path, skip_edge_encoder: bool = True):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = extract_state_dict(payload)
    target_state = model.state_dict()

    loaded_state = {}
    loaded_keys = []
    skipped_keys = []

    for raw_key, tensor in source_state.items():
        key = raw_key
        if key.startswith("encoder."):
            key = key[len("encoder.") :]

        if skip_edge_encoder and key.startswith("edge_encoder."):
            skipped_keys.append(
                {
                    "source_key": raw_key,
                    "mapped_key": key,
                    "source_shape": tuple(tensor.shape),
                    "target_shape": "skipped_by_flag",
                }
            )
            continue

        if key in target_state and target_state[key].shape == tensor.shape:
            loaded_state[key] = tensor
            loaded_keys.append((raw_key, key, tuple(tensor.shape)))
        else:
            skipped_keys.append(
                {
                    "source_key": raw_key,
                    "mapped_key": key,
                    "source_shape": tuple(tensor.shape) if hasattr(tensor, "shape") else None,
                    "target_shape": tuple(target_state[key].shape) if key in target_state else None,
                }
            )

    if not loaded_state:
        print("[WARN] No compatible pretrained encoder weights were loaded.")
        return {
            "checkpoint_path": str(checkpoint_path),
            "loaded_param_count": 0,
            "loaded_keys": [],
            "skipped_param_count": len(skipped_keys),
            "skipped_keys_preview": skipped_keys[:10],
        }

    target_state.update(loaded_state)
    model.load_state_dict(target_state, strict=False)
    print(
        f"Loaded {len(loaded_keys)} compatible pretrained parameter tensors from: "
        f"{checkpoint_path}"
    )
    for raw_key, mapped_key, shape in loaded_keys[:10]:
        print(f"  [loaded] {raw_key} -> {mapped_key} shape={shape}")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} pretrained tensors (showing up to 10).")
        for item in skipped_keys[:10]:
            print(f"  [skipped] {item}")

    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_param_count": len(loaded_keys),
        "loaded_keys": [mapped_key for _, mapped_key, _ in loaded_keys],
        "skipped_param_count": len(skipped_keys),
        "skipped_keys_preview": skipped_keys[:10],
    }


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
    if monitor_name == "val_f1":
        return float(threshold_df["f1"].max())
    if monitor_name == "val_dual_score":
        return float(threshold_df["dual_score"].max())
    if monitor_name == "val_pr_auc":
        return float(val_metrics["pr_auc"])
    if monitor_name == "val_auc":
        return float(val_metrics["roc_auc"])
    if monitor_name == "val_loss":
        return float(val_metrics["loss"])
    raise ValueError(f"Unsupported monitor: {monitor_name}")


def build_output_dirs(output_dir: Path):
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "thresholds": output_dir / "thresholds",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


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


def confusion_matrix_to_dict(cm):
    if cm is None:
        return None
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


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


def mean_std(vals):
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


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
    best_threshold = None
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
        val_prob = val_raw["probs"]
        val_labels = val_raw["labels"]

        selected_threshold, threshold_df, selected_row = select_threshold(
            val_labels, val_prob, metric_name=threshold_metric
        )
        monitor_value = get_monitor_value(monitor, val_raw, threshold_df)
        scheduler.step(monitor_value)
        lr = float(optimizer.param_groups[0]["lr"])

        epoch_record = {
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
        epoch_records.append(epoch_record)
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
            f"f1score={selected_row['f1']:.4f} | mcc={selected_row['mcc']:.4f}"
        )

        if improved:
            best_value = float(monitor_value)
            best_epoch = int(epoch)
            best_threshold = float(selected_threshold)
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
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "epochs_completed": len(history_rows),
    }


def run_single_seed(args, model_cls, train_list, val_list, test_list, seed: int, run_dir: Path):
    output_dirs = build_output_dirs(run_dir)
    set_seed(seed)

    train_loader, val_loader, test_loader = build_loaders_with_seed(
        train_list, val_list, test_list, batch_size=args.batch_size, seed=seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(
        in_dim=train_list[0].x.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        edge_attr_dim=train_list[0].edge_attr.size(1),
    )
    pretrained_load_info = None
    if args.pretrained_encoder_path:
        pretrained_path = Path(args.pretrained_encoder_path).expanduser().resolve()
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"pretrained_encoder_path does not exist: {pretrained_path}"
            )
        pretrained_load_info = load_pretrained_encoder_weights(
            model,
            pretrained_path,
            skip_edge_encoder=args.skip_pretrained_edge_encoder,
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

    val_metrics = evaluate_split(
        y_true=val_raw["labels"],
        y_prob=val_raw["probs"],
        loss=val_raw["loss"],
        threshold=selected_threshold,
    )
    test_metrics = evaluate_split(
        y_true=test_raw["labels"],
        y_prob=test_raw["probs"],
        loss=test_raw["loss"],
        threshold=selected_threshold,
    )

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
        "pretrained_encoder_path": None
        if pretrained_load_info is None
        else pretrained_load_info["checkpoint_path"],
        "pretrained_loaded_param_count": 0
        if pretrained_load_info is None
        else int(pretrained_load_info["loaded_param_count"]),
        "pretrained_loaded_keys": []
        if pretrained_load_info is None
        else pretrained_load_info["loaded_keys"],
        "pretrained_skipped_param_count": 0
        if pretrained_load_info is None
        else int(pretrained_load_info["skipped_param_count"]),
        "pretrained_skipped_keys_preview": []
        if pretrained_load_info is None
        else pretrained_load_info["skipped_keys_preview"],
        "skip_pretrained_edge_encoder": bool(args.skip_pretrained_edge_encoder),
        "best_epoch_metrics": train_summary["best_metrics"],
        "train_samples": int(len(train_list)),
        "val_samples": int(len(val_list)),
        "test_samples": int(len(test_list)),
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

    train_list = torch.load(args.train_pt, map_location="cpu", weights_only=False)["data_list"]
    val_list = torch.load(args.val_pt, map_location="cpu", weights_only=False)["data_list"]
    test_list = torch.load(args.test_pt, map_location="cpu", weights_only=False)["data_list"]

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
        result = run_single_seed(args, model_cls, train_list, val_list, test_list, seed, run_dir)
        results.append(result)

    if len(results) > 1:
        summarize_multi_seed(results, output_dir)
        print(f"\nSaved multi-seed summary to {output_dir}")


if __name__ == "__main__":
    main()
