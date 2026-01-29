import os
import csv
import argparse
import math

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import ConvVAE, vae_loss_bce_logits
from .viz import save_image_grid, save_recon_pairs_grid, save_latent_interpolation


def set_seed(seed: int):
    """
    Make runs more reproducible.
    Note: full determinism is not guaranteed across all backends (CUDA/DirectML/CPU),
    but this stabilizes typical experimentation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto"):
    """
    Backend-robust device selection.

    - CUDA (NVIDIA, and some AMD Linux builds) if available
    - MPS (Apple) if available
    - DirectML (Windows AMD/Intel) if torch-directml is installed
    - CPU otherwise

    This keeps your code portable: only the environment changes.
    """
    prefer = prefer.lower()

    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    if prefer in ("auto", "directml", "dml"):
        try:
            import torch_directml
            return torch_directml.device()
        except Exception as e:
            raise RuntimeError(f"DirectML requested but not available: {repr(e)}")

    return torch.device("cpu")


def move_optimizer_to_device(optimizer, device):
    """
    After loading an optimizer state_dict (often stored on CPU),
    move any tensor states to the current device so training can resume
    on CUDA/DirectML without device mismatch errors.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


@torch.no_grad()
def save_recons_and_samples(model, x_batch, device, out_dir, epoch, z_dim):
    """
    Save:
      - reconstructions (original + reconstructed)
      - random samples from z ~ N(0,I)
    """
    model.eval()

    x = x_batch.to(device)
    logits, _, _ = model(x)
    x_hat = torch.sigmoid(logits)  # convert logits -> probabilities for viewing

    # Recon grid: originals paired with reconstructions (explicit layout)
    cols = 8
    pair_rows = 4  # shows 32 pairs; set to 8 to show 64 pairs
    pairs = min(x.size(0), cols * pair_rows)

    save_recon_pairs_grid(
        x[:pairs],
        x_hat[:pairs],
        os.path.join(out_dir, f"recon_epoch{epoch:03d}.png"),
        cols=cols,
        pair_rows=pair_rows,
        title="Originals (odd rows) vs Reconstructions (even rows)"
    )

    # Sample grid: z ~ N(0,I)
    z = torch.randn(64, z_dim, device=device)
    samp_logits = model.decode_logits(z)
    samp = torch.sigmoid(samp_logits)
    save_image_grid(
        samp,
        os.path.join(out_dir, f"samples_epoch{epoch:03d}.png"),
        n=8,
        title="Samples (z ~ N(0,I))"
    )

def _get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def write_run_meta(args, device, out_dir: str) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "args": vars(args),
    }

    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta_warmup_epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "directml", "mps"])
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Windows
    ap.add_argument("--out_dir", type=str, default="artifacts")

    # --- Saving / checkpointing options (mutually exclusive) ---
    save_group = ap.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save_best",
        action="store_true",
        help="Save only the best model (overwrites best.pt when metric improves)."
    )
    save_group.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save periodic checkpoints (vae_epochXXX.pt) to ckpt_dir."
    )

    # Best-model selection options (used only with --save_best)
    ap.add_argument("--best_metric", type=str, default="total",
                    choices=["total", "recon", "kl"],
                    help="Metric to minimize when selecting the best epoch.")
    ap.add_argument("--best_split", type=str, default="test",
                    choices=["test", "train"],
                    help="Split used for selecting best epoch.")
    ap.add_argument("--best_path", type=str, default="",
                    help="Optional explicit path for best checkpoint. Default: <out_dir>/best.pt")

    # Periodic checkpoint options (used only with --save_checkpoints)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints",
                    help="Directory to write periodic checkpoints.")
    ap.add_argument("--save_every", type=int, default=5,
                    help="Save a checkpoint every N epochs (only with --save_checkpoints).")
    # Resume works with either mode (best.pt or vae_epochXXX.pt)
    ap.add_argument("--resume", type=str, default="",
                    help="Path to a checkpoint to resume from (best.pt or vae_epochXXX.pt).")

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    write_run_meta(args=args, device=device, out_dir=args.out_dir)

    # Prepare saving paths / state
    best_path = args.best_path if args.best_path else os.path.join(args.out_dir, "best.pt")
    best_value = math.inf
    best_epoch = -1

    if args.save_checkpoints:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Using device:", device)

    # Fashion-MNIST images are 28x28 grayscale; ToTensor() gives float in [0,1]
    tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)

    # pin_memory helps mainly for CUDA host->device copies; keep it conservative
    dev_type = getattr(device, "type", "")
    pin_memory = (dev_type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    model = ConvVAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        move_optimizer_to_device(opt, device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1

        # If resuming from a best checkpoint that stored these fields, restore them
        if "best_value" in ckpt:
            best_value = float(ckpt["best_value"])
        if "best_epoch" in ckpt:
            best_epoch = int(ckpt["best_epoch"])

        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Metrics file (append-friendly)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

    # Fixed batch for consistent visuals across epochs
    x_vis, _ = next(iter(test_loader))

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "split", "total", "recon", "kl"])

        for epoch in range(start_epoch, args.epochs):
            # Dynamic Beta
            warm = max(1, args.beta_warmup_epochs)
            beta_eff = args.beta * min(1.0, float(epoch + 1) / float(warm))

            # -------- Train --------
            model.train()
            tr_tot = tr_rec = tr_kl = 0.0
            n_tr = 0

            for x, _ in train_loader:
                x = x.to(device)

                logits, mu, logvar = model(x)
                loss, recon, kl = vae_loss_bce_logits(x, logits, mu, logvar, beta=beta_eff)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                tr_tot += loss.item() * bs
                tr_rec += recon.item() * bs
                tr_kl += kl.item() * bs
                n_tr += bs

            tr_tot /= n_tr
            tr_rec /= n_tr
            tr_kl /= n_tr
            writer.writerow([epoch, "train", tr_tot, tr_rec, tr_kl])
            f.flush()

            # -------- Test --------
            model.eval()
            te_tot = te_rec = te_kl = 0.0
            n_te = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    logits, mu, logvar = model(x)
                    loss, recon, kl = vae_loss_bce_logits(x, logits, mu, logvar, beta=beta_eff)

                    bs = x.size(0)
                    te_tot += loss.item() * bs
                    te_rec += recon.item() * bs
                    te_kl += kl.item() * bs
                    n_te += bs

            te_tot /= n_te
            te_rec /= n_te
            te_kl /= n_te
            writer.writerow([epoch, "test", te_tot, te_rec, te_kl])
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train total={tr_tot:.2f} recon={tr_rec:.2f} kl={tr_kl:.2f} | "
                f"test total={te_tot:.2f} recon={te_rec:.2f} kl={te_kl:.2f}"
            )

            save_recons_and_samples(
                model,
                x_vis,
                device,
                args.out_dir,
                epoch,
                args.z_dim
            )
            save_latent_interpolation(
                model,
                x_vis,
                device,
                out_path=os.path.join(args.out_dir, f"interp_epoch{epoch:03d}.png"),
                steps = 12
            )

            # -------------------------
            # Save best OR save checkpoints (mutually exclusive)
            # -------------------------
            if args.save_best:
                if args.best_split == "test":
                    metric_map = {"total": te_tot, "recon": te_rec, "kl": te_kl}
                else:
                    metric_map = {"total": tr_tot, "recon": tr_rec, "kl": tr_kl}

                candidate = float(metric_map[args.best_metric])

                if candidate < best_value:
                    best_value = candidate
                    best_epoch = epoch

                    torch.save(
                        {
                            "epoch": epoch,
                            "best_value": best_value,
                            "best_epoch": best_epoch,
                            "best_metric": args.best_metric,
                            "best_split": args.best_split,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "args": vars(args),
                            "train": {"total": tr_tot, "recon": tr_rec, "kl": tr_kl},
                            "test": {"total": te_tot, "recon": te_rec, "kl": te_kl},
                        },
                        best_path
                    )
                    print(
                        f"[best] saved {best_path} | epoch={best_epoch} {args.best_split}.{args.best_metric}={best_value:.4f}")

            elif args.save_checkpoints:
                should_save = ((epoch + 1) % args.save_every == 0) or (epoch == args.epochs - 1)
                if should_save:
                    ckpt_path = os.path.join(args.ckpt_dir, f"vae_epoch{epoch:03d}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "args": vars(args),
                            "train": {"total": tr_tot, "recon": tr_rec, "kl": tr_kl},
                            "test": {"total": te_tot, "recon": te_rec, "kl": te_kl},
                        },
                        ckpt_path
                    )
                    print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
