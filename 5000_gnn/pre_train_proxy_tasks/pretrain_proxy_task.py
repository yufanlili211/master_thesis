import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_AA_FEATURE_CSV = (
    PROJECT_ROOT / "data" / "rdkit_descriptors" / "68_aa_node_features_mapping.csv"
)
DEFAULT_EMBEDDING_PT = PROJECT_ROOT / "data" / "chemberta" / "peptide_embeddings.pt"
DEFAULT_EMBEDDING_INDEX_CSV = (
    PROJECT_ROOT / "data" / "chemberta" / "peptide_embeddings_index.csv"
)
DEFAULT_RAW_XLSX = (
    PROJECT_ROOT
    / "data"
    / "raw_data"
    / "canya_data_sampled_4993_smiles_deduplicated_length_over2.xlsx"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining for peptide GNN encoder with attribute masking and ChemBERTa alignment."
    )
    parser.add_argument("--aa_feature_csv", type=str, default=str(DEFAULT_AA_FEATURE_CSV))
    parser.add_argument("--embedding_pt", type=str, default=str(DEFAULT_EMBEDDING_PT))
    parser.add_argument(
        "--embedding_index_csv", type=str, default=str(DEFAULT_EMBEDDING_INDEX_CSV)
    )
    parser.add_argument(
        "--raw_xlsx",
        type=str,
        default=str(DEFAULT_RAW_XLSX),
        help="Optional fallback source of aa_seq if the index CSV is unavailable.",
    )
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--pooling", choices=["mean", "add"], default="mean")
    parser.add_argument("--edge_attr_dim", type=int, default=2)
    parser.add_argument("--max_hop", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--align_loss", choices=["mse", "cosine"], default="mse")
    parser.add_argument(
        "--align_weight",
        type=float,
        default=100.0,
        help=(
            "Weight for ChemBERTa alignment loss. Default 100.0 makes alignment "
            "contribute meaningfully when raw align MSE is much smaller than mask MSE."
        ),
    )
    parser.add_argument("--mask_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aa_feature_map(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "1-Letter" not in df.columns:
        raise ValueError("AA feature CSV must contain a '1-Letter' column.")

    feature_cols = [col for col in df.columns if col != "1-Letter"]
    aa_to_feature = {}
    aa_to_idx = {}
    for _, row in df.iterrows():
        aa = str(row["1-Letter"]).strip()
        aa_to_idx[aa] = len(aa_to_idx)
        aa_to_feature[aa] = torch.tensor(
            row[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32
        )
    return aa_to_feature, aa_to_idx, feature_cols


def load_embedding_table(index_csv: Path, raw_xlsx: Path):
    if index_csv.exists():
        df = pd.read_csv(index_csv)
    elif raw_xlsx.exists():
        df = pd.read_excel(raw_xlsx)
        if "aa_seq" not in df.columns:
            raise ValueError("Fallback raw XLSX must contain an 'aa_seq' column.")
        df = df.loc[:, ["aa_seq"]].copy()
        df["embedding_idx"] = np.arange(len(df))
    else:
        raise FileNotFoundError(
            f"Neither embedding index CSV nor fallback XLSX exists: {index_csv}, {raw_xlsx}"
        )

    required_cols = {"aa_seq", "embedding_idx"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Embedding index file is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["aa_seq", "embedding_idx"]).copy()
    df["aa_seq"] = df["aa_seq"].astype(str).str.strip()
    df["embedding_idx"] = df["embedding_idx"].astype(int)
    return df


def build_linear_edge_index(num_nodes: int, max_hop: int):
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    src_list = []
    dst_list = []
    for hop in range(1, max_hop + 1):
        if num_nodes <= hop:
            continue
        src = torch.arange(0, num_nodes - hop, dtype=torch.long)
        dst = torch.arange(hop, num_nodes, dtype=torch.long)
        src_list.extend([src, dst])
        dst_list.extend([dst, src])

    edge_index = torch.stack([torch.cat(src_list, dim=0), torch.cat(dst_list, dim=0)], dim=0)
    return edge_index


def build_linear_edge_attr(num_nodes: int, edge_attr_dim: int, max_hop: int):
    if edge_attr_dim != 2:
        raise ValueError(
            f"edge_attr_dim must be 2 to match the main GINE model, but got {edge_attr_dim}."
        )
    if num_nodes <= 1:
        return torch.empty((0, edge_attr_dim), dtype=torch.float32)

    edge_attr_list = []
    for hop in range(1, max_hop + 1):
        if num_nodes <= hop:
            continue
        num_edges = num_nodes - hop
        hop_value = float(hop)
        forward_attr = torch.tensor([1.0, hop_value], dtype=torch.float32).repeat(num_edges, 1)
        backward_attr = torch.tensor([-1.0, hop_value], dtype=torch.float32).repeat(num_edges, 1)
        edge_attr_list.extend([forward_attr, backward_attr])
    return torch.cat(edge_attr_list, dim=0)


def _make_gine_mlp(hidden_dim: int):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class PeptideMaskingDataset(Dataset):
    """
    Builds a line graph from each peptide sequence and applies node attribute masking on-the-fly.
    """

    def __init__(
        self,
        index_df,
        embeddings,
        aa_to_feature,
        aa_to_idx,
        mask_ratio: float = 0.15,
        edge_attr_dim: int = 2,
        max_hop: int = 2,
        dynamic_mask: bool = True,
        seed: int = 42,
    ):
        self.mask_ratio = mask_ratio
        self.edge_attr_dim = edge_attr_dim
        self.max_hop = max_hop
        self.dynamic_mask = dynamic_mask
        self.seed = seed
        self.samples = []
        self.num_skipped = 0

        for row in index_df.itertuples(index=False):
            aa_seq = str(row.aa_seq).strip()
            embedding_idx = int(row.embedding_idx)

            if embedding_idx < 0 or embedding_idx >= len(embeddings):
                self.num_skipped += 1
                continue

            unknown = [aa for aa in aa_seq if aa not in aa_to_feature]
            if unknown:
                self.num_skipped += 1
                continue

            x = torch.stack([aa_to_feature[aa].clone() for aa in aa_seq], dim=0)
            aa_target = torch.tensor([aa_to_idx[aa] for aa in aa_seq], dtype=torch.long)
            edge_index = build_linear_edge_index(len(aa_seq), self.max_hop)
            edge_attr = build_linear_edge_attr(len(aa_seq), self.edge_attr_dim, self.max_hop)
            chemberta_embedding = embeddings[embedding_idx].float().clone().view(1, -1)

            self.samples.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    aa_target=aa_target,
                    aa_seq=aa_seq,
                    chemberta_embedding=chemberta_embedding,
                )
            )

        if not self.samples:
            raise ValueError("No valid peptide samples were built for pretraining.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx].clone()
        num_nodes = data.x.size(0)

        num_mask = max(1, int(round(num_nodes * self.mask_ratio)))
        if self.dynamic_mask:
            perm = torch.randperm(num_nodes)
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            perm = torch.randperm(num_nodes, generator=generator)
        mask_idx = perm[:num_mask]

        original_x = data.x.clone()
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[mask_idx] = True

        data.x[mask] = 0.0
        data.mask = mask
        data.mask_target = original_x
        return data


class GINEVirtualNodeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        edge_attr_dim: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pooling not in ["mean", "add"]:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_dim)
        self.dropout = dropout
        self.pooling = pooling
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GINEConv(_make_gine_mlp(hidden_dim)))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.vn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)
        e = self.edge_encoder(edge_attr)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        virtualnode_emb = x.new_zeros((num_graphs, self.hidden_dim))

        for i in range(self.num_layers):
            x = x + virtualnode_emb[batch]
            x = self.convs[i](x, edge_index, e)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i != self.num_layers - 1:
                vn_update = global_add_pool(x, batch)
                virtualnode_emb = virtualnode_emb + self.vn_mlp(vn_update)

        if self.pooling == "add":
            graph_emb = global_add_pool(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)
        return x, graph_emb


class PretrainNet(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        num_aa_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        chemberta_dim: int,
        pooling: str = "mean",
        edge_attr_dim: int = 2,
    ):
        super().__init__()
        self.encoder = GINEVirtualNodeEncoder(
            in_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            edge_attr_dim=edge_attr_dim,
        )
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_aa_classes),
        )
        self.alignment_projector = nn.Linear(hidden_dim, chemberta_dim)

    def forward(self, data):
        node_emb, graph_emb = self.encoder(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        mask_pred = self.mask_predictor(node_emb)
        align_pred = self.alignment_projector(graph_emb)
        return {
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "mask_pred": mask_pred,
            "align_pred": align_pred,
        }


def compute_losses(outputs, batch, align_loss: str, mask_weight: float, align_weight: float):
    mask = batch.mask.bool()
    if mask.sum().item() == 0:
        loss_mask = outputs["mask_pred"].sum() * 0.0
    else:
        loss_mask = F.cross_entropy(outputs["mask_pred"][mask], batch.aa_target[mask])

    target_embedding = batch.chemberta_embedding.view(outputs["align_pred"].size(0), -1)
    if align_loss == "mse":
        loss_align = F.mse_loss(outputs["align_pred"], target_embedding)
    else:
        cosine = F.cosine_similarity(outputs["align_pred"], target_embedding, dim=-1)
        loss_align = 1.0 - cosine.mean()

    total_loss = mask_weight * loss_mask + align_weight * loss_align
    return total_loss, loss_mask.detach(), loss_align.detach()


def run_epoch(loader, model, optimizer, device, align_loss, mask_weight, align_weight, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mask_loss = 0.0
    total_align_loss = 0.0
    total_graphs = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss, loss_mask, loss_align = compute_losses(
                outputs,
                batch,
                align_loss=align_loss,
                mask_weight=mask_weight,
                align_weight=align_weight,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_graphs = batch.num_graphs
            total_graphs += num_graphs
            total_loss += loss.item() * num_graphs
            total_mask_loss += loss_mask.item() * num_graphs
            total_align_loss += loss_align.item() * num_graphs

    denom = max(total_graphs, 1)
    return {
        "loss": total_loss / denom,
        "loss_mask": total_mask_loss / denom,
        "loss_align": total_align_loss / denom,
    }


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aa_feature_csv = Path(args.aa_feature_csv).expanduser().resolve()
    embedding_pt = Path(args.embedding_pt).expanduser().resolve()
    embedding_index_csv = Path(args.embedding_index_csv).expanduser().resolve()
    raw_xlsx = Path(args.raw_xlsx).expanduser().resolve()

    aa_to_feature, aa_to_idx, feature_cols = load_aa_feature_map(aa_feature_csv)
    embedding_payload = torch.load(embedding_pt, map_location="cpu", weights_only=False)
    embeddings = embedding_payload["embeddings"].float()
    index_df = load_embedding_table(embedding_index_csv, raw_xlsx)

    full_dataset = PeptideMaskingDataset(
        index_df=index_df,
        embeddings=embeddings,
        aa_to_feature=aa_to_feature,
        aa_to_idx=aa_to_idx,
        mask_ratio=args.mask_ratio,
        edge_attr_dim=args.edge_attr_dim,
        max_hop=args.max_hop,
        dynamic_mask=True,
        seed=args.seed,
    )

    if args.val_ratio > 0 and len(full_dataset) > 1:
        val_size = max(1, int(len(full_dataset) * args.val_ratio))
        train_size = len(full_dataset) - val_size
        if train_size == 0:
            train_size = len(full_dataset) - 1
            val_size = 1
        generator = torch.Generator().manual_seed(args.seed)
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        train_df = index_df.iloc[train_indices].reset_index(drop=True)
        val_df = index_df.iloc[val_indices].reset_index(drop=True)
        train_dataset = PeptideMaskingDataset(
            index_df=train_df,
            embeddings=embeddings,
            aa_to_feature=aa_to_feature,
            aa_to_idx=aa_to_idx,
            mask_ratio=args.mask_ratio,
            edge_attr_dim=args.edge_attr_dim,
            max_hop=args.max_hop,
            dynamic_mask=True,
            seed=args.seed,
        )
        val_dataset = PeptideMaskingDataset(
            index_df=val_df,
            embeddings=embeddings,
            aa_to_feature=aa_to_feature,
            aa_to_idx=aa_to_idx,
            mask_ratio=args.mask_ratio,
            edge_attr_dim=args.edge_attr_dim,
            max_hop=args.max_hop,
            dynamic_mask=False,
            seed=args.seed,
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainNet(
        node_input_dim=len(feature_cols),
        num_aa_classes=len(aa_to_idx),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        chemberta_dim=int(embeddings.shape[1]),
        pooling=args.pooling,
        edge_attr_dim=args.edge_attr_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_metric = float("inf")
    best_encoder_path = output_dir / "best_gnn_encoder.pt"

    metadata = {
        "num_samples": len(full_dataset),
        "train_samples": len(train_dataset),
        "val_samples": 0 if val_dataset is None else len(val_dataset),
        "num_skipped": full_dataset.num_skipped,
        "node_feature_dim": len(feature_cols),
        "num_aa_classes": len(aa_to_idx),
        "chemberta_dim": int(embeddings.shape[1]),
        "encoder_type": "gine_virtual_node",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "edge_attr_dim": args.edge_attr_dim,
        "max_hop": args.max_hop,
        "mask_ratio": args.mask_ratio,
        "mask_task": "masked_amino_acid_classification",
        "align_loss": args.align_loss,
    }
    save_json(output_dir / "run_config.json", {**vars(args), **metadata})

    print(
        f"Built dataset with {len(full_dataset)} samples | "
        f"train={len(train_dataset)} | val={0 if val_dataset is None else len(val_dataset)} | "
        f"skipped={full_dataset.num_skipped} | "
        f"feature_dim={len(feature_cols)} | chemberta_dim={int(embeddings.shape[1])}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            align_loss=args.align_loss,
            mask_weight=args.mask_weight,
            align_weight=args.align_weight,
            train=True,
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                val_loader,
                model,
                optimizer,
                device,
                align_loss=args.align_loss,
                mask_weight=args.mask_weight,
                align_weight=args.align_weight,
                train=False,
            )
            monitor_value = val_metrics["loss"]
        else:
            val_metrics = None
            monitor_value = train_metrics["loss"]

        if monitor_value < best_metric:
            best_metric = monitor_value
            torch.save(model.encoder.state_dict(), best_encoder_path)

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_loss_mask": float(train_metrics["loss_mask"]),
            "train_loss_align": float(train_metrics["loss_align"]),
            "best_monitor_loss": float(best_metric),
        }
        if val_metrics is not None:
            row.update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_loss_mask": float(val_metrics["loss_mask"]),
                    "val_loss_align": float(val_metrics["loss_align"]),
                }
            )
        history.append(row)

        if val_metrics is None:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"mask={train_metrics['loss_mask']:.6f} | "
                f"align={train_metrics['loss_align']:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"train_mask={train_metrics['loss_mask']:.6f} | "
                f"train_align={train_metrics['loss_align']:.6f} | "
                f"val_mask={val_metrics['loss_mask']:.6f} | "
                f"val_align={val_metrics['loss_align']:.6f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "pretrain_history.tsv", sep="\t", index=False)
    save_json(
        output_dir / "summary.json",
        {
            **metadata,
            "best_encoder_path": str(best_encoder_path),
            "best_monitor_loss": float(best_metric),
        },
    )
    print(f"Saved encoder weights to: {best_encoder_path}")


if __name__ == "__main__":
    main()
