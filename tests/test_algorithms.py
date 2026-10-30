import numpy as np
import torch
from fedcare.algorithms import FedCARE, FedNova, weighted_average_states
from fedcare.models import CompactCNN


def _states(n=3, seed=0):
    torch.manual_seed(seed)
    return [CompactCNN(3, 4, 4).state_dict() for _ in range(n)]


def test_weighted_average_states():
    s = _states()
    w = [0.2, 0.3, 0.5]
    out = weighted_average_states(s, w)
    k = "head.weight"
    assert torch.allclose(out[k], sum(wi * si[k] for wi, si in zip(w, s)), atol=1e-6)


def test_fednova_equals_fedavg_without_momentum_and_equal_steps():
    torch.manual_seed(0)
    g = CompactCNN(3, 4, 4)
    states = _states(3, 1)
    res = [{"n": n, "steps": 7, "state": s} for n, s in zip([10, 20, 30], states)]
    algo = FedNova({"momentum": 0.0}, 4, 16, 3)
    algo.aggregate(g, res)
    ref = weighted_average_states(states, np.array([10, 20, 30]) / 60)
    for k, v in g.state_dict().items():
        if torch.is_floating_point(v):
            assert torch.allclose(v, ref[k], atol=1e-5), k


def test_fedcare_class_aware_head_aggregation():
    C = 4
    g = CompactCNN(3, C, 4)
    states = _states(2, 2)
    counts = [np.array([10, 0, 5, 0]), np.array([0, 10, 5, 0])]
    algo = FedCARE({}, C, 16, 2)
    res = [{"k": i, "n": int(c.sum()), "counts": c, "steps": 1, "state": s, "payload": {}}
           for i, (c, s) in enumerate(zip(counts, states))]
    algo.aggregate(g, res)
    W = g.state_dict()["head.weight"]
    assert torch.allclose(W[0], states[0]["head.weight"][0])          # only client 0 has class 0
    assert torch.allclose(W[1], states[1]["head.weight"][1])          # only client 1 has class 1
    assert torch.allclose(W[2], 0.5 * (states[0]["head.weight"][2] + states[1]["head.weight"][2]), atol=1e-6)
    sw = np.array([15, 15]) / 30                                     # class absent everywhere: sample weights
    assert torch.allclose(W[3], sum(wi * s["head.weight"][3] for wi, s in zip(sw, states)), atol=1e-6)
