"""
download_checkpoints.py — Download pre-trained model checkpoints.

Functionality:
    Downloads and verifies all required pre-trained model checkpoints:
    - MACE-MP-0 (universal NNP)
    - CHGNet (universal NNP)
    - M3GNet (universal NNP)
    - DiffCSP++ (learned score baseline)
    - MatterGen (learned score baseline)
    - CDVAE (data-driven baseline)

    Verifies: file hash (MD5/SHA256), file size, model load test

Usage:
    python scripts/download_checkpoints.py --all
    python scripts/download_checkpoints.py --model mace chgnet
    python scripts/download_checkpoints.py --model diffcsp --output-dir checkpoints/

Arguments:
    --all                  Download all available models
    --model NAME           Download specific model(s) (repeatable)
    --output-dir PATH      Directory to save checkpoints (default: checkpoints/)
    --verify-only          Verify existing checkpoints without downloading
"""
