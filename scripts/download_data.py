#!/usr/bin/env python
"""Download the official MedMNIST v2 (28x28) files and verify their MD5 checksums.

  pip install medmnist
  python scripts/download_data.py --datasets bloodmnist dermamnist pathmnist
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedcare.data import MEDMNIST_MD5, md5sum  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["bloodmnist", "dermamnist"])
ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "data"))
a = ap.parse_args()
os.makedirs(a.root, exist_ok=True)
import medmnist  # noqa: E402
from medmnist import INFO  # noqa: E402
for name in a.datasets:
    cls = getattr(medmnist, INFO[name]["python_class"])
    cls(split="train", download=True, root=a.root, size=28)
    path = os.path.join(a.root, f"{name}.npz")
    if not os.path.exists(path):                       # older/newer medmnist naming
        alt = os.path.join(a.root, f"{name}_28.npz")
        if os.path.exists(alt):
            os.replace(alt, path)
    got = md5sum(path)
    print(name, "OK" if got == MEDMNIST_MD5[name] else f"MD5 MISMATCH ({got})")
