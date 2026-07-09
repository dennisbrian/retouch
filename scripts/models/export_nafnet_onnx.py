#!/usr/bin/env python3
"""
Export NAFNet-SIDD-width32 PyTorch model to ONNX format for retouch denoise pipeline.

This script:
1. Downloads NAFNet-SIDD-width32.pth from HuggingFace (or uses cached copy)
2. Builds NAFNet model with correct architecture
3. Exports to ONNX with dynamic H/W axes
4. Runs parity checks on multiple input shapes
5. Outputs model hash and size for manifest

Setup (run in a separate venv with torch):
    python3 -m venv /tmp/nafnet_export_venv
    /tmp/nafnet_export_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/nafnet_export_venv/bin/pip install onnx onnxruntime numpy

    # On macOS, torch can be installed directly:
    pip install torch onnx onnxruntime numpy

Run:
    /tmp/nafnet_export_venv/bin/python3 export_nafnet_onnx.py [--pth PATH] [--out PATH]
"""

import sys
import os
import hashlib
import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

# Import the vendored NAFNet architecture
from nafnet_arch import NAFNet


def download_pth(url, dest_path):
    """Download file with streaming and .part rename pattern."""
    dest = Path(dest_path)
    part_path = dest.with_suffix(dest.suffix + '.part')

    print(f"Downloading {url} -> {dest}")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
        )
        with urllib.request.urlopen(req) as response, open(part_path, 'wb') as out:
            size = response.headers.get('content-length')
            if size:
                size = int(size)
                print(f"  Total: {size / 1e6:.1f} MB")

            downloaded = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if size:
                    pct = 100 * downloaded / size
                    print(f"  {pct:.0f}%", end='\r')

        part_path.rename(dest)
        print(f"Saved to {dest}")
        return str(dest)
    except Exception as e:
        if part_path.exists():
            part_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def sha256_of_file(path, chunk_size=65536):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_nafnet_model():
    """Build NAFNet-SIDD-width32 architecture."""
    model = NAFNet(
        img_channel=3,
        width=32,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2]
    )
    return model


def load_checkpoint(model, pth_path):
    """Load checkpoint into model. Handle 'params' key wrapper."""
    print(f"Loading checkpoint from {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu')

    # Handle 'params' key wrapper (common in some checkpoints)
    if isinstance(ckpt, dict) and 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Checkpoint loaded successfully")


def export_onnx(model, output_path, opset_version=17):
    """Export model to ONNX with dynamic H/W axes."""
    print(f"Exporting to ONNX: {output_path}")

    # Dummy input: (batch=1, channels=3, H=256, W=256)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    print(f"ONNX exported to {output_path}")


def parity_check(model, onnx_path):
    """Run parity checks between torch and onnxruntime on multiple shapes."""
    print("\nRunning parity checks...")

    # Create ONNX Runtime session
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    test_shapes = [
        (1, 3, 256, 256),    # Standard
        (1, 3, 512, 512),    # Large
        (1, 3, 320, 480),    # Non-square
        (1, 3, 180, 240),    # Sub-tile
    ]

    model.eval()
    max_diff_overall = 0.0

    with torch.no_grad():
        for shape in test_shapes:
            # Random input in [0, 1] range
            inp = torch.rand(*shape, dtype=torch.float32)

            # PyTorch inference
            torch_out = model(inp).numpy()

            # ONNX inference
            onnx_out = sess.run(None, {'input': inp.numpy()})[0]

            # Compute max absolute difference
            diff = np.abs(torch_out - onnx_out).max()
            max_diff_overall = max(max_diff_overall, diff)

            print(f"  Shape {shape}: max_diff = {diff:.6e}")

    print(f"Overall max diff: {max_diff_overall:.6e}")

    if max_diff_overall >= 1e-3:
        raise RuntimeError(f"Parity check failed: max_diff {max_diff_overall} >= 1e-3")

    print("Parity check passed!")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pth', type=str, default=None,
        help='Path to NAFNet-SIDD-width32.pth. If not provided, downloads to /tmp/nafnet_src/'
    )
    parser.add_argument(
        '--out', type=str, default='/tmp/nafnet_src/nafnet_denoise.onnx',
        help='Output ONNX path (default: /tmp/nafnet_src/nafnet_denoise.onnx)'
    )

    args = parser.parse_args()

    # Ensure /tmp/nafnet_src exists
    os.makedirs('/tmp/nafnet_src', exist_ok=True)

    # Resolve .pth path
    if args.pth:
        pth_path = args.pth
        if not Path(pth_path).exists():
            raise FileNotFoundError(f"--pth {pth_path} not found")
    else:
        pth_path = '/tmp/nafnet_src/NAFNet-SIDD-width32.pth'
        if not Path(pth_path).exists():
            # Download from HuggingFace
            url = 'https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width32.pth'
            pth_path = download_pth(url, pth_path)

    # Build and load model
    print("Building NAFNet model...")
    model = build_nafnet_model()
    load_checkpoint(model, pth_path)

    # Export to ONNX
    export_onnx(model, args.out)

    # Run parity checks
    parity_check(model, args.out)

    # Compute and print hash
    h = sha256_of_file(args.out)
    size_bytes = Path(args.out).stat().st_size

    print(f"\nModel exported successfully!")
    print(f"  Path: {args.out}")
    print(f"  SHA256: {h}")
    print(f"  Size: {size_bytes} bytes ({size_bytes / 1e6:.2f} MB)")
    print(f"\nFor manifest.json, use:")
    print(f'  "sha256": "{h}",')
    print(f'  "size_bytes": {size_bytes}')


if __name__ == '__main__':
    main()
