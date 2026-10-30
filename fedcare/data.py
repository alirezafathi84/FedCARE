"""Data loading, federated partitioning and acquisition-shift simulation.

All datasets are the official MedMNIST v2 28x28 ``.npz`` files. Every file is
verified against the MD5 checksum published in the ``medmnist`` package, so a
run can never silently use a modified copy.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# MD5 checksums of the official 28x28 files (medmnist==3.0.2, medmnist.info.INFO)
MEDMNIST_MD5 = {
    "pathmnist": "a8b06965200029087d5bd730944a56c1",
    "dermamnist": "0744692d530f8e62ec473284d019b0c7",
    "bloodmnist": "7053d0359d879ad8a5505303e11de1dc",
    "organamnist": "68e3f8846a6bd62f0c9bf841c0d9eacc",
    "organcmnist": "b9ceb9546e10131b32923c5bbeaea2b1",
    "organsmnist": "9ab87b696fb54e2a387ebe992d6ed5f1",
}

_ORGAN = ["bladder", "femur-L", "femur-R", "heart", "kidney-L", "kidney-R",
          "liver", "lung-L", "lung-R", "pancreas", "spleen"]
CLASS_NAMES = {
    "pathmnist": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "bloodmnist": ["BAS", "EOS", "ERB", "IG", "LYM", "MON", "NEU", "PLT"],
    "organamnist": _ORGAN, "organcmnist": _ORGAN, "organsmnist": _ORGAN,
}

# Flips/rotations are label-preserving for histology, dermoscopy and blood
# smears, but NOT for CT slices (left/right organs), so they are opt-in.
DEFAULT_AUGMENT = {
    "pathmnist": ("hflip", "vflip", "rot90"),
    "dermamnist": ("hflip", "vflip"),
    "bloodmnist": ("hflip", "vflip", "rot90"),
    "organamnist": (), "organcmnist": (), "organsmnist": (),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_medmnist(name: str, root: str, verify: bool = True) -> Dict[str, np.ndarray]:
    """Load ``<root>/<name>.npz`` and return uint8 NHWC images and int64 labels."""
    path = os.path.join(root, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py --datasets {name}`.")
    if verify and name in MEDMNIST_MD5:
        got = md5sum(path)
        if got != MEDMNIST_MD5[name]:
            raise ValueError(f"MD5 mismatch for {path}: {got} != {MEDMNIST_MD5[name]}")
    raw = np.load(path)
    out = {}
    for split in ("train", "val", "test"):
        x = raw[f"{split}_images"]
        if x.ndim == 3:                       # grayscale -> NHWC with one channel
            x = x[..., None]
        out[f"{split}_x"] = x.astype(np.uint8)
        out[f"{split}_y"] = raw[f"{split}_labels"].reshape(-1).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# Sampling and partitioning
# --------------------------------------------------------------------------- #
def stratified_subsample(y: np.ndarray, n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """Class-stratified subsample of size ~n (at least one sample per class)."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    alloc = np.maximum(1, np.round(counts / counts.sum() * n).astype(int))
    idx = [rng.choice(np.flatnonzero(y == c), size=min(a, cnt), replace=False)
           for c, a, cnt in zip(classes, alloc, counts)]
    return np.sort(np.concatenate(idx))


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float,
                        rng: np.random.Generator, min_size: int = 10,
                        max_tries: int = 1000) -> List[np.ndarray]:
    """Label-skew partition of Hsu et al. (2019) with the NIID-Bench balancing rule.

    For each class a proportion vector p ~ Dir(alpha * 1_K) splits the class
    indices among clients; clients that already hold more than N/K samples get
    no further share. Re-sampled until every client holds >= ``min_size``.
    """
    n, classes = len(y), np.unique(y)
    for _ in range(max_tries):
        buckets: List[list] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.flatnonzero(y == c)
            rng.shuffle(idx_c)
            p = rng.dirichlet(np.full(num_clients, alpha))
            p = np.array([pk * (len(b) < n / num_clients) for pk, b in zip(p, buckets)])
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [rng.permutation(np.asarray(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError("Could not draw a Dirichlet partition satisfying min_size.")


# --------------------------------------------------------------------------- #
# Acquisition (site) shift for histology: HED stain perturbation + optics
# --------------------------------------------------------------------------- #
# Ruifrok & Johnston stain vectors (same convention as skimage.color.rgb2hed)
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
_LOG_ADJ = np.log(1e-6)


def rgb2hed(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb, 1e-6)
    return np.maximum((np.log(rgb) / _LOG_ADJ) @ HED_FROM_RGB, 0.0)


def hed2rgb(hed: np.ndarray) -> np.ndarray:
    return np.clip(np.exp((hed * _LOG_ADJ) @ RGB_FROM_HED), 0.0, 1.0)


def sample_site_params(rng: np.random.Generator, strength: float = 1.0) -> dict:
    """Static per-site acquisition parameters (stain, exposure, optics, sensor noise)."""
    s = strength
    return {
        "hed_scale": rng.uniform(1 - 0.35 * s, 1 + 0.35 * s, size=3).round(4).tolist(),
        "hed_shift": rng.uniform(-0.04 * s, 0.04 * s, size=3).round(4).tolist(),
        "brightness": float(np.round(rng.uniform(1 - 0.10 * s, 1 + 0.10 * s), 4)),
        "contrast": float(np.round(rng.uniform(1 - 0.15 * s, 1 + 0.15 * s), 4)),
        "blur_sigma": float(np.round(rng.uniform(0.0, 0.9 * s), 4)),
        "noise_std": float(np.round(rng.uniform(0.0, 0.03 * s), 4)),
    }


def apply_site_shift(images: np.ndarray, p: dict, rng: np.random.Generator,
                     mode: str = "hed") -> np.ndarray:
    """Apply one site's acquisition parameters to uint8 NHWC RGB images.

    ``mode="hed"``: H&E stain jitter in Ruifrok-Johnston HED space (histology,
    following Tellez et al., 2019). ``mode="od"``: the same jitter applied per
    RGB channel in optical-density space (Beer-Lambert), a stain-agnostic model
    of staining-intensity differences (e.g. May-Gruenwald-Giemsa blood smears).
    """
    x = images.astype(np.float64) / 255.0
    if mode == "hed":
        hed = rgb2hed(x) * np.asarray(p["hed_scale"]) + np.asarray(p["hed_shift"])
        x = hed2rgb(np.maximum(hed, 0.0))
    elif mode == "od":
        od = -np.log(np.maximum(x, 1e-3))
        od = np.maximum(od * np.asarray(p["hed_scale"]) + np.asarray(p["hed_shift"]) * 10, 0.0)
        x = np.exp(-od)
    else:
        raise ValueError(mode)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    x = ((x - mean) * p["contrast"] + mean) * p["brightness"]
    if p["blur_sigma"] > 0.05:
        x = gaussian_filter(x, sigma=(0, p["blur_sigma"], p["blur_sigma"], 0))
    if p["noise_std"] > 0:
        x = x + rng.normal(0.0, p["noise_std"], size=x.shape)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Federated settings
# --------------------------------------------------------------------------- #
def build_setting(cfg: dict, seed: int, data_root: str) -> dict:
    """Materialise a federated setting.

    Returns a dict with ``clients`` (list of {x, y, domain}), ``val`` ({x, y}),
    ``tests`` (name -> {x, y}), ``num_classes``, ``in_ch``, ``class_names``,
    ``augment`` and ``meta`` (partition statistics, site parameters).
    The same ``seed`` always yields the same subsample, partition and site
    parameters, so all methods are compared on identical federations.
    """
    rng = np.random.default_rng(seed)
    kind = cfg.get("partition", "dirichlet")
    meta: dict = {"seed": seed, "kind": kind}

    if kind == "dirichlet":
        name = cfg["dataset"]
        d = load_medmnist(name, data_root, verify=cfg.get("verify_md5", True))
        sub = stratified_subsample(d["train_y"], cfg.get("n_train"), rng)
        xtr, ytr = d["train_x"][sub], d["train_y"][sub]
        parts = dirichlet_partition(ytr, cfg["num_clients"], cfg["alpha"], rng,
                                    cfg.get("min_client_size", 10))
        clients = [{"x": xtr[p], "y": ytr[p], "domain": 0} for p in parts]
        if cfg.get("site_shift") in ("stain", "od"):
            mode = "hed" if cfg["site_shift"] == "stain" else "od"
            site_params = []
            for k, c in enumerate(clients):
                p = sample_site_params(rng, cfg.get("shift_strength", 1.0))
                c["x"] = apply_site_shift(c["x"], p, rng, mode)
                site_params.append(p)
            meta["site_params"] = site_params
        vsub = stratified_subsample(d["val_y"], cfg.get("n_val"), rng)
        val = {"x": d["val_x"][vsub], "y": d["val_y"][vsub]}
        tests = {"test": {"x": d["test_x"], "y": d["test_y"]}}
        num_classes = len(CLASS_NAMES[name])
        class_names = CLASS_NAMES[name]
        augment = tuple(cfg.get("augment", DEFAULT_AUGMENT.get(name, ())))
        meta["train_class_counts"] = np.bincount(ytr, minlength=num_classes).tolist()

    elif kind == "multidomain":
        # Each domain is a separate MedMNIST file sharing one label space
        # (e.g. OrganA/C/S = axial/coronal/sagittal CT planes).
        clients, vals, tests = [], [], {}
        names = cfg["domains"]
        for di, name in enumerate(names):
            d = load_medmnist(name, data_root, verify=cfg.get("verify_md5", True))
            sub = stratified_subsample(d["train_y"], cfg.get("n_train_per_domain"), rng)
            xtr, ytr = d["train_x"][sub], d["train_y"][sub]
            parts = dirichlet_partition(ytr, cfg["clients_per_domain"], cfg["alpha"], rng,
                                        cfg.get("min_client_size", 10))
            clients += [{"x": xtr[p], "y": ytr[p], "domain": di} for p in parts]
            vsub = stratified_subsample(d["val_y"], cfg.get("n_val_per_domain"), rng)
            vals.append((d["val_x"][vsub], d["val_y"][vsub]))
            tsub = stratified_subsample(d["test_y"], cfg.get("n_test_per_domain"), rng)
            tests[name] = {"x": d["test_x"][tsub], "y": d["test_y"][tsub]}
        val = {"x": np.concatenate([v[0] for v in vals]), "y": np.concatenate([v[1] for v in vals])}
        num_classes = len(CLASS_NAMES[names[0]])
        class_names = CLASS_NAMES[names[0]]
        augment = tuple(cfg.get("augment", ()))
        allc = np.concatenate([c["y"] for c in clients])
        meta["train_class_counts"] = np.bincount(allc, minlength=num_classes).tolist()
        meta["domains"] = list(names)
    else:
        raise ValueError(f"Unknown partition kind {kind!r}")

    meta["client_sizes"] = [int(len(c["y"])) for c in clients]
    meta["client_class_counts"] = [np.bincount(c["y"], minlength=num_classes).tolist()
                                   for c in clients]
    return {"clients": clients, "val": val, "tests": tests, "num_classes": num_classes,
            "in_ch": int(clients[0]["x"].shape[-1]), "class_names": class_names,
            "augment": augment, "meta": meta}


# --------------------------------------------------------------------------- #
# Tensors and augmentation
# --------------------------------------------------------------------------- #
def to_tensor(x: np.ndarray, channels_last: bool = False) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW in [-1, 1]."""
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    return t.contiguous(memory_format=torch.channels_last) if channels_last else t.contiguous()


def augment_batch(x: torch.Tensor, ops, gen: torch.Generator) -> torch.Tensor:
    """Random label-preserving flips / 90-degree rotations (per sample)."""
    if not ops:
        return x
    b = x.shape[0]
    if "hflip" in ops:
        m = torch.rand(b, generator=gen) < 0.5
        x[m] = x[m].flip(3)
    if "vflip" in ops:
        m = torch.rand(b, generator=gen) < 0.5
        x[m] = x[m].flip(2)
    if "rot90" in ops:
        ks = torch.randint(0, 4, (b,), generator=gen)
        for k in (1, 2, 3):
            m = ks == k
            if m.any():
                x[m] = torch.rot90(x[m], k, (2, 3))
    return x
