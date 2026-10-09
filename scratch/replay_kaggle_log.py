"""Counterfactual (open-loop) replay of a logged BDSF-AFL run through JointDecisionEngine.

Feeds every row of a ``*_updates.csv`` (the evidence the server logged at decision time) back
through the decision engine, in order, once with LEGACY settings and once with the PATCHED ones,
and compares the decisions per honest / Byzantine client.

What is held fixed (open loop): the logged evidence itself -- reference vector, subspace metrics,
reputation I_i/P_i, behavioural history, version lag. So effects that need a re-run (rejected
attackers leaving the reference buffer, reputation slashing feeding back into the Top-K reference,
the client's own TRS history changing) are NOT captured. Per-client consecutive-downweight counts
and the engine's own suspicion/pinning ledgers ARE evolved from the replayed decisions.

If the CSV has no ``sim_global_raw`` column (runs before the raw-cosine patch) the pinning gates see
the post-subspace cosine, which is rescaled every round and hides S2's constant-cosine signature, so
attacker detection from the pinning gates is UNDER-stated in that case.

    python scratch/replay_kaggle_log.py path/to/run_updates.csv [--theta-cos 0.15] [--jitter 0.04]
"""
import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from server.decision_engine import JointDecisionEngine  # noqa: E402
from shared.types import TemporalEvidence, SpatialEvidence, BehavioralEvidence  # noqa: E402

LEGACY_OVERRIDES = dict(honest_cert_enabled=False, pin_probation_enabled=False, pin_mass_min=0,
                        staleness_exponent=0.5, staleness_floor=0.0)


def _f(x: str) -> Optional[float]:
    return None if x in ("", None) else float(x)


def reconstruct_raw_cos(r: dict) -> Optional[float]:
    """Approximate the pre-subspace ("raw") cosine from logged subspace metrics.

    Exact when the reference lies in span(Q) (true right after the basis is refreshed from it):
        v_clean = v_par + w * v_perp,  w = w_damp * w_temporal,  |v_par| / |v| = rho_parallel
        cos_raw = cos_filtered * |v_clean| / |v| = cos_filtered * sqrt(rho^2 + w^2 (1 - rho^2))
    Validated on the S2 log: recovered values pile up at theta_cos + 0.05 = 0.2000 (+-0.003).
    """
    f = _f(r.get("sim_global", ""))
    rho = _f(r.get("subspace_rho_parallel", ""))
    wd, wt = _f(r.get("subspace_w_damp", "")), _f(r.get("subspace_w_temporal", ""))
    if f is None or rho is None or wd is None or wt is None:
        return None
    w = wd * wt
    return float(f * np.sqrt(rho ** 2 + w ** 2 * max(0.0, 1.0 - rho ** 2)))


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "INFO" or r.get("client_id") in ("", None):
                continue
            rows.append(r)
    rows.sort(key=lambda r: int(float(r["round"])))
    return rows


def load_cfg(over: dict) -> dict:
    with open(os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(over)
    return cfg


def replay(rows: List[dict], cfg: dict, use_reconstructed_raw: bool = False) -> List[dict]:
    eng = JointDecisionEngine(cfg)
    c_dw: Dict[int, int] = {}
    out = []
    for r in rows:
        cid = int(r["client_id"])
        rnd = int(float(r["round"]))
        # Rows the engine never saw (rejected by the subspace gate in the aggregator) are passed through.
        if r["reason"] in ("MACRO_MANIFOLD_INVERSION", "CROSS_LAYER_COLLAPSE", "SUBSPACE_ENERGY_COLLAPSE"):
            out.append(dict(src=r, status=r["status"], reason=r["reason"], weight=0.0, engine=False, diag={}))
            continue
        lag = int(float(r["version_lag"] or 0))
        temp = TemporalEvidence(
            g_i=float(r["g_i"] or 0.0), lower_fence=_f(r["lower_fence"]), upper_fence=_f(r["upper_fence"]),
            fence_margin=float(r["fence_margin"] or 0.0), client_z_score=0.0, is_burn_in=False,
            temporal_mature=(r["temporal_mature"] == "True"), version_lag=lag)
        spat = SpatialEvidence(
            sim_global=_f(r["sim_global"]), norm_raw=float(r["norm_raw"] or 0.0),
            norm_clipped=float(r["norm_clipped"] or 0.0), norm_ratio_median=_f(r["norm_ratio_median"]),
            dynamic_bound_C=_f(r["dynamic_bound_C"]), spatial_mature=(r["spatial_mature"] == "True"),
            spatial_coherence=float(r["spatial_coherence"] or 0.0),
            prc_score=_f(r["prc_score"]), tra_score=_f(r["tra_score"]),
            sim_global_raw=(reconstruct_raw_cos(r) if use_reconstructed_raw else _f(r.get("sim_global_raw", ""))))
        beh = BehavioralEvidence(
            sim_self_mean=None, sim_self_max=_f(r["sim_self_max"]), history_depth=int(float(r["history_depth"] or 0)),
            sim_anchor=_f(r["sim_anchor"]), sim_frozen_anchor=_f(r["sim_frozen_anchor"]),
            anchor_drift=_f(r["anchor_drift"]), consecutive_dw=c_dw.get(cid, 0),
            gdv_score=_f(r["gdv_score"]), dbp_score=_f(r["dbp_score"]), trs_score=_f(r["trs_score"]),
            behavioral_mature=(r["behavioral_mature"] == "True"))
        o = eng.evaluate(cid, temp, spat, beh, float(r["I_i"] or 1.0), float(r["P_i"] or 1.0),
                         current_round=rnd, version_lag=lag,
                         subspace_mu=_f(r["subspace_mu"]), subspace_rho_parallel=_f(r["subspace_rho_parallel"]))
        if o.action == "DOWNWEIGHT":
            if not o.diagnostic_features.get("pin_suspect"):
                c_dw[cid] = c_dw.get(cid, 0) + 1
        elif o.action == "ACCEPT":
            c_dw[cid] = 0
        out.append(dict(src=r, status=o.action, reason=o.primary_reason, weight=o.aggregation_weight, engine=True, diag=o.diagnostic_features))
    return out


def summarize(res: List[dict]) -> Dict[str, Dict[str, float]]:
    post = [x for x in res if x["src"]["is_warmup"] != "True"]
    out = {}
    for tag, flag in (("honest", "False"), ("byz", "True")):
        rs = [x for x in post if x["src"]["is_byzantine"] == flag]
        n = max(1, len(rs))
        dw = lambda x: x["weight"] * np.sqrt(1.0 + float(x["src"]["version_lag"] or 0))
        out[tag] = dict(
            n=len(rs),
            full_accept=100.0 * sum(x["status"] == "ACCEPT" for x in rs) / n,
            downweight=100.0 * sum(x["status"] == "DOWNWEIGHT" for x in rs) / n,
            reject=100.0 * sum(x["status"] in ("REJECT", "QUARANTINE") for x in rs) / n,
            admitted=100.0 * sum(x["status"] in ("ACCEPT", "DOWNWEIGHT") for x in rs) / n,
            mean_weight_logged=float(np.sum([x["weight"] for x in rs]) / n),
            mean_weight_decision_only=float(np.sum([dw(x) for x in rs]) / n),
        )
    return out


def is_pinned_proxy(x: dict, target: float, tol: float = 0.005) -> bool:
    """Byzantine row whose reconstructed raw cosine sits on the S2 boundary value (theta_cos + margin):
    a proxy for 'S2 actually modified this update' (S2 leaves updates untouched when cos < target)."""
    if x["src"]["is_byzantine"] != "True":
        return False
    c = reconstruct_raw_cos(x["src"])
    return c is not None and abs(c - target) <= tol


def split_summary(res: List[dict], target: float) -> Dict[str, Dict[str, float]]:
    post = [x for x in res if x["src"]["is_warmup"] != "True"]
    groups = {
        "honest": [x for x in post if x["src"]["is_byzantine"] == "False"],
        "byz_pinned(proxy: S2-modified)": [x for x in post if is_pinned_proxy(x, target)],
        "byz_other(unmodified, honest-like)": [x for x in post if x["src"]["is_byzantine"] == "True" and not is_pinned_proxy(x, target)],
    }
    out = {}
    for k, rs in groups.items():
        n = max(1, len(rs))
        dw = lambda x: x["weight"] * np.sqrt(1.0 + float(x["src"]["version_lag"] or 0))
        out[k] = dict(n=len(rs), full=100.0 * sum(x["status"] == "ACCEPT" for x in rs) / n,
                      dwn=100.0 * sum(x["status"] == "DOWNWEIGHT" for x in rs) / n,
                      rej=100.0 * sum(x["status"] in ("REJECT", "QUARANTINE") for x in rs) / n,
                      w=float(np.sum([dw(x) for x in rs]) / n))
    return out


def fidelity(res: List[dict]) -> float:
    post = [x for x in res if x["src"]["is_warmup"] != "True"]
    return 100.0 * sum(x["status"] == x["src"]["status"] for x in post) / max(1, len(post))


def reasons(res: List[dict], byz: bool) -> Dict[str, int]:
    post = [x for x in res if x["src"]["is_warmup"] != "True" and (x["src"]["is_byzantine"] == "True") == byz]
    return dict(Counter(f'{x["status"]}:{x["reason"]}' for x in post).most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--theta-cos", type=float, default=None)
    ap.add_argument("--jitter", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value overrides for the PATCHED engine")
    ap.add_argument("--theta-cos-attack", type=float, default=0.20,
                    help="boundary value the S2 attacker targets (theta_cos + s2_cosine_margin) for the pinned proxy")
    ap.add_argument("--reconstruct-raw", action="store_true",
                    help="CSV has no sim_global_raw: approximate it from the logged subspace metrics (both replays)")
    a = ap.parse_args()
    rows = load_rows(a.csv)
    common = {}
    if a.theta_cos is not None:
        common["theta_cos"] = a.theta_cos
    if a.jitter is not None:
        common["stochastic_jitter_max"] = a.jitter
    patched_over = dict(common)
    for it in a.set:
        k, v = it.split("=", 1)
        patched_over[k] = yaml.safe_load(v)
    if a.reconstruct_raw and "pin_mass_tol" not in patched_over:
        # reconstructed cosines carry ~1e-3 error (exact raw logs do not): loosen the mass-point tolerance
        patched_over["pin_mass_tol"] = 0.004
    legacy = replay(rows, load_cfg({**common, **LEGACY_OVERRIDES}), a.reconstruct_raw)
    patched = replay(rows, load_cfg(patched_over), a.reconstruct_raw)
    print(f"rows={len(rows)}  has sim_global_raw column: {'sim_global_raw' in rows[0]}")
    print(f"legacy-engine replay agrees with the logged action on {fidelity(legacy):.1f}% of post-warmup rows")
    for name, res in (("LOGGED/legacy replay", legacy), ("PATCHED replay", patched)):
        s = summarize(res)
        print(f"\n== {name}")
        for tag in ("honest", "byz"):
            m = s[tag]
            print(f"  {tag:6s} n={m['n']:3d} full={m['full_accept']:5.1f}% dw={m['downweight']:5.1f}% rej={m['reject']:5.1f}% "
                  f"admitted={m['admitted']:5.1f}%  mean w (logged scale)={m['mean_weight_logged']:.4f}  decision-only={m['mean_weight_decision_only']:.3f}")
        for k, m in split_summary(res, a.theta_cos_attack).items():
            print(f"    {k:36s} n={m['n']:3d} full={m['full']:5.1f}% dw={m['dwn']:5.1f}% rej={m['rej']:5.1f}%  mean decision weight={m['w']:.3f}")
        pr = [x for x in res if x["src"]["is_warmup"] != "True" and x["reason"] == "BOUNDARY_PROBATION_DOWNWEIGHT" and x["src"]["is_byzantine"] == "False"]
        if pr:
            print(f"    honest rows on probation: {len(pr)}  (thin record only: {sum(1 for x in pr if not x['diag'].get('pin_suspect'))}, suspect: {sum(1 for x in pr if x['diag'].get('pin_suspect'))})")
        print("  honest reasons:", reasons(res, False))
        print("  byz reasons:   ", reasons(res, True))


if __name__ == "__main__":
    main()
