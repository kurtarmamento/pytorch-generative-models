# Results — ConvVAE on Fashion-MNIST

## Configuration (run summary)
- Dataset: Fashion-MNIST (28×28 grayscale)
- Architecture: 2-layer conv encoder + MLP latent heads + transpose-conv decoder
- Latent dimension: 32
- Objective: BCEWithLogits (pixel-summed) + beta * KL
- Beta schedule: warm-up to beta=1.0 over 5 epochs
- Optimizer: Adam, lr=2e-4
- Batch size: 128
- Epochs: 10–20
- Device: (CPU / CUDA / DirectML)

## Training log excerpt (example)
Epoch 000 | train total=337.29 recon=325.63 kl=11.65 | test total=278.92 recon=263.82 kl=15.09  
Epoch 001 | train total=270.59 recon=255.46 kl=15.12 | test total=268.53 recon=253.25 kl=15.27  
Epoch 002 | train total=263.61 recon=248.63 kl=14.99 | test total=262.61 recon=247.53 kl=15.08  
Epoch 003 | train total=259.00 recon=244.21 kl=14.79 | test total=258.74 recon=244.18 kl=14.56  
Epoch 004 | train total=255.37 recon=240.70 kl=14.66 | test total=255.81 recon=241.10 kl=14.70  
Epoch 005 | train total=252.51 recon=238.05 kl=14.47 | test total=252.79 recon=238.63 kl=14.16  
Epoch 006 | train total=250.29 recon=235.97 kl=14.32 | test total=251.57 recon=237.21 kl=14.37  
Epoch 007 | train total=248.55 recon=234.36 kl=14.19 | test total=249.42 recon=235.21 kl=14.22  
Epoch 008 | train total=247.10 recon=232.92 kl=14.18 | test total=248.17 recon=233.94 kl=14.23  
Epoch 009 | train total=246.02 recon=231.83 kl=14.18 | test total=247.20 recon=233.25 kl=13.95  

Notes:
- `recon` decreased substantially over the first 10 epochs, indicating improved reconstruction quality.
- `kl` remained non-zero (~14–15), indicating the latent variable is being used (no immediate KL collapse).

## Visual artifacts
Generated during training (see `artifacts/`):
- `recon_epoch009.png` — paired originals and reconstructions
- `samples_epoch009.png` — unconditional samples from `z ~ N(0, I)`
- `interp_epoch009.png` — latent interpolation between two inputs (smoothness indicates structured latent space)

## Qualitative observations
- Reconstructions become sharper and more class-consistent with training.
- Samples remain blurrier than typical GAN outputs (expected behavior for a basic VAE).
- Interpolations are smooth, suggesting the latent space is continuous and meaningful.

## Best checkpoint
If `--save_best` was used:
- `artifacts/best.pt` stores the best epoch under the chosen selection metric (e.g., lowest test total).


### For sample outputs refer to example folder