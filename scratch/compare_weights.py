"""Open-loop comparison of aggregation-weight configurations on a logged run (see replay_kaggle_log.py caveats).
    python scratch/compare_weights.py run_updates.csv
Weight columns use the engine's OWN aggregation_weight (includes whichever staleness mode is configured)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from replay_kaggle_log import load_rows, load_cfg, replay, is_pinned_proxy

PREV = dict(staleness_relative=False, p4_manifold_weight=0.0, straggler_shock_min_clients=0)
CONFIGS = {
    "prev patch (abs stale .5)":          PREV,
    "abs stale exp 0.25":                 dict(PREV, staleness_exponent=0.25),
    "relative stale":                     dict(PREV, staleness_relative=True),
    "manifold boost .6 only":             dict(PREV, p4_manifold_weight=0.6),
    "shock exempt only":                  dict(PREV, straggler_shock_min_clients=2),
    "NEW (rel + boost .6 + shock)":       dict(staleness_relative=True, p4_manifold_weight=0.6, straggler_shock_min_clients=2),
    "NEW + boost .8":                     dict(staleness_relative=True, p4_manifold_weight=0.8, straggler_shock_min_clients=2),
}

def stats(res):
    post = [x for x in res if x["src"]["is_warmup"] != "True" and x["src"]["status"] in ("ACCEPT", "DOWNWEIGHT", "REJECT")]
    H = [x for x in post if x["src"]["is_byzantine"] == "False"]
    B = [x for x in post if x["src"]["is_byzantine"] == "True"]
    w = lambda xs: float(sum(x["weight"] or 0.0 for x in xs if x["status"] != "REJECT"))
    pin = [x for x in B if is_pinned_proxy(x, 0.20)]
    oth = [x for x in B if not is_pinned_proxy(x, 0.20)]
    full = lambda xs: np.mean([x["status"] == "ACCEPT" for x in xs])
    rej = lambda xs: np.mean([x["status"] == "REJECT" for x in xs])
    wh, wb = w(H), w(B)
    return dict(honest_rej=rej(H), honest_full=full(H), honest_mean_w=wh / len(H), byz_w_share=wb / (wh + wb),
                pinned_rej=rej(pin), pinned_w=w(pin), byz_other_w=w(oth), n_pin=len(pin))

if __name__ == "__main__":
    rows = load_rows(sys.argv[1])
    print(f"{'config':34s} {'h_rej':>6s} {'h_full':>6s} {'h_mean_w':>8s} {'byz_w%':>7s} {'pin_rej':>7s} {'pin_w':>6s} {'byz_oth_w':>9s}")
    for name, over in CONFIGS.items():
        s = stats(replay(rows, load_cfg(over)))
        print(f"{name:34s} {s['honest_rej']:6.3f} {s['honest_full']:6.3f} {s['honest_mean_w']:8.3f} {100*s['byz_w_share']:6.1f}% {s['pinned_rej']:7.3f} {s['pinned_w']:6.2f} {s['byz_other_w']:9.2f}")
