"""End-to-end run of every method on a tiny synthetic MedMNIST-format dataset."""
import numpy as np
import pytest
from fedcare.algorithms import ALGORITHMS
from fedcare.runner import Runner


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    def split(n):
        y = rng.integers(0, 4, n)
        x = (rng.uniform(0, 1, (n, 28, 28, 3)) * 60 + y[:, None, None, None] * 45).astype(np.uint8)
        return x, y[:, None]
    tr, va, te = split(400), split(80), split(120)
    np.savez(root / "toy.npz", train_images=tr[0], train_labels=tr[1], val_images=va[0],
             val_labels=va[1], test_images=te[0], test_labels=te[1])
    return str(root)


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_every_method_runs(method, data_root, monkeypatch):
    import fedcare.data as D
    monkeypatch.setitem(D.CLASS_NAMES, "toy", ["a", "b", "c", "d"])
    setting = {"partition": "dirichlet", "dataset": "toy", "num_clients": 4, "alpha": 0.3,
               "n_train": None, "n_val": None, "site_shift": "stain", "verify_md5": False}
    train = {"model": "cnn", "width": 4, "rounds": 2, "local_epochs": 1, "batch_size": 32,
             "lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4, "eval_every": 1,
             "test_subsample": None, "channels_last": False, "threads": 1}
    res = Runner(setting, train, method, {}, 0, data_root, log=lambda m: None).run()
    fin = res["final"][method]["test"]
    for k in ("acc", "bacc", "macro_f1", "macro_auc", "worst_recall"):
        assert 0.0 <= fin[k] <= 1.0
    assert len(fin["per_class"]["f1"]) == 4 and len(fin["confusion"]) == 4
    assert len(res["history"]) == 2
    if method == "fedavg":
        assert {"fedavg+ccvr", "fedavg+lrcal"} <= set(res["final"])
    if method == "fedcare":
        assert "fedcare+lrcal" in res["final"]
        assert res["extra_upload_floats_per_client_round"] == 4          # label counts only


def test_fedcare_with_subspace_alignment_runs(data_root, monkeypatch):
    import fedcare.data as D
    monkeypatch.setitem(D.CLASS_NAMES, "toy", ["a", "b", "c", "d"])
    setting = {"partition": "dirichlet", "dataset": "toy", "num_clients": 4, "alpha": 0.3,
               "n_train": None, "n_val": None, "verify_md5": False}
    train = {"model": "cnn", "width": 4, "rounds": 3, "local_epochs": 1, "batch_size": 32,
             "lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4, "eval_every": 1,
             "test_subsample": None, "channels_last": False, "threads": 1}
    res = Runner(setting, train, "fedcare", {"beta": 0.5, "rank": 2}, 0, data_root, log=lambda m: None).run()
    assert res["extra_upload_floats_per_client_round"] > 4
    assert 0.0 <= res["final"]["fedcare"]["test"]["macro_f1"] <= 1.0
