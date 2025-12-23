import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvVAE, vae_loss_bce_logits
from viz import save_image_grid, save_recon_pairs_grid


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
        except Exception:
            pass

    return torch.device("cpu")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "directml", "mps"])
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Windows
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

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

    # Metrics file (append-friendly)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

    # Fixed batch for consistent visuals across epochs
    x_vis, _ = next(iter(test_loader))

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "split", "total", "recon", "kl"])

        for epoch in range(args.epochs):
            # -------- Train --------
            model.train()
            tr_tot = tr_rec = tr_kl = 0.0
            n_tr = 0

            for x, _ in train_loader:
                x = x.to(device)

                logits, mu, logvar = model(x)
                loss, recon, kl = vae_loss_bce_logits(x, logits, mu, logvar, beta=args.beta)

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
                    loss, recon, kl = vae_loss_bce_logits(x, logits, mu, logvar, beta=args.beta)

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

            save_recons_and_samples(model, x_vis, device, args.out_dir, epoch, args.z_dim)


if __name__ == "__main__":
    main()
