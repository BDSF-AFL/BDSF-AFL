"""Synthetic replay of the BDSF-AFL server pipeline (no CIFAR-10 / no ResNet needed).

Purpose
-------
Drive the REAL ``AggregatorServer`` (subspace engine, spatial validator, behavioural memory,
joint decision engine, reputation manager) and the REAL ``AttackInjector`` with synthetic
client updates, so decision-engine changes can be A/B compared in seconds on a laptop.

What is synthetic
-----------------
Only the *honest client gradients*. They are generated from a low-rank "class manifold" model
(shared global direction + Dirichlet class mix + persistent per-client direction + fresh noise)
and calibrated so that the honest population shows the statistics reported from the Kaggle run
(raw cosine to reference ~0.14-0.2, TRS ~0.9, mu ~0.25, rho_parallel ~0.85). Absolute accuracy
cannot be read from this; the *relative* admit / downweight / reject behaviour of honest vs
Byzantine updates can.

Usage
-----
    python scratch/synthetic_replay.py --attack COMPOUND --seeds 0 1 2
    python scratch/synthetic_replay.py --attack ADAPTIVE --set cert_enabled=false
"""
import argparse
import copy
import heapq
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch  # noqa: E402  (real torch, or the numpy shim when torch is unavailable)

from server.aggregator import AggregatorServer  # noqa: E402
from simulation.attack_injector import AttackInjector  # noqa: E402
from shared.types import UpdateSubmission  # noqa: E402


# --------------------------------------------------------------------------------------
# Honest client model
# --------------------------------------------------------------------------------------
class HonestWorld:
    """Low-rank synthetic world producing honest updates for N clients."""

    def __init__(self, N: int, D: int, seed: int, w_g=0.05, w_k=0.45, w_s=0.40, w_n=0.10,
                 drift=0.03, n_classes=10, dirichlet=0.5):
        self.rng = np.random.default_rng(seed)
        self.N, self.D = N, D
        self.w = (w_g, w_k, w_s, w_n)
        self.drift = drift
        self.g = self._unit(self.rng.standard_normal(D))
        self.E = np.stack([self._unit(self.rng.standard_normal(D)) for _ in range(n_classes)])
        self.p = self.rng.dirichlet(np.full(n_classes, dirichlet), size=N)
        self.s = np.stack([self._unit(self.rng.standard_normal(D)) for _ in range(N)])

    @staticmethod
    def _unit(x):
        n = np.linalg.norm(x)
        return x / (n + 1e-12)

    def _drift_dirs(self):
        d = self.drift
        self.g = self._unit((1 - d) * self.g + d * self._unit(self.rng.standard_normal(self.D)))

    def honest_update(self, cid: int, step: int) -> np.ndarray:
        if step % self.N == 0:
            self._drift_dirs()
        d = self.drift
        self.s[cid] = self._unit((1 - d) * self.s[cid] + d * self._unit(self.rng.standard_normal(self.D)))
        K = self._unit((np.sqrt(self.p[cid])[:, None] * self.E).sum(axis=0))
        n = self._unit(self.rng.standard_normal(self.D))
        wg, wk, ws, wn = self.w
        v = np.sqrt(wg) * self.g + np.sqrt(wk) * K + np.sqrt(ws) * self.s[cid] + np.sqrt(wn) * n
        norm = max(0.05, self.rng.normal(1.0, 0.08))
        return (self._unit(v) * norm).astype(np.float32)


# --------------------------------------------------------------------------------------
# Logger stub (the aggregator only needs .log_update(**kw))
# --------------------------------------------------------------------------------------
class ListLogger:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def log_update(self, **kw):
        self.rows.append(dict(kw))

    def __getattr__(self, name):  # tolerate any other logger call
        return lambda *a, **k: None


# --------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------
def load_config(overrides: Optional[Dict[str, Any]] = None) -> dict:
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["log_dir"] = "logs/synthetic_replay/"
    if overrides:
        cfg.update(overrides)
    return cfg


def run_replay(cfg_over: Optional[Dict[str, Any]] = None, attack: str = "COMPOUND", seed: int = 0,
               D: int = 3000, n_updates: int = 800, N: int = 20, n_byz: int = 4,
               world_kwargs: Optional[dict] = None, time_sigma: float = 0.25) -> Dict[str, Any]:
    cfg = load_config(cfg_over)
    cfg["N_clients"] = N
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1000)
    world = HonestWorld(N, D, seed, **(world_kwargs or {}))
    byz_ids = set(int(i) for i in rng.choice(N, size=n_byz, replace=False))

    logger = ListLogger()
    W0 = torch.tensor(np.zeros(D, dtype=np.float32), dtype=torch.float32)
    server = AggregatorServer(cfg, W0, list(range(N)), logger)
    injectors = {}
    for cid in byz_ids:
        server.register_client_ground_truth(cid, is_byzantine=True)
        injectors[cid] = AttackInjector(attack, cid, cfg)

    period = rng.lognormal(0.0, 0.15, size=N)
    last_update_time = np.zeros(N)
    pulled_version = np.zeros(N, dtype=int)
    heap = []
    for cid in range(N):
        heapq.heappush(heap, (float(period[cid] * rng.lognormal(0, time_sigma)), cid))

    step = 0
    while step < n_updates and heap:
        t_honest, cid = heapq.heappop(heap)
        honest = world.honest_update(cid, step)
        honest_t = torch.tensor(honest, dtype=torch.float32)
        gap_honest = t_honest - last_update_time[cid]
        if cid in byz_ids:
            hist = server.temporal_filter.gap_history
            ctx = {
                "honest_g_i": float(gap_honest),
                "median_g": float(np.median(hist)) if hist else None,
                "W_global": server.get_global_weights(),
                "ref_delta_W": server.spatial_validator._build_reference(),
                "theta_cos": server.spatial_validator.theta_cos,
                "own_P_i": server.rep_manager.get(cid)[1],
            }
            dW, mod_g = injectors[cid].inject(honest_t, ctx)
            t_sub = float(last_update_time[cid] + mod_g)
            modified = bool(np.linalg.norm(np.asarray(dW, dtype=np.float64) - honest.astype(np.float64)) > 1e-6 * max(1.0, np.linalg.norm(honest)))
        else:
            dW, t_sub = honest_t, float(t_honest)
            modified = False

        sub = UpdateSubmission(client_id=cid, delta_W=dW, t_submit=t_sub, tau=t_honest,
                               model_version_at_pull=int(pulled_version[cid]))
        n_before = len(logger.rows)
        resp = server.handle_update(sub)
        for r in logger.rows[n_before:]:
            if r.get("client_id") == cid and "attack_modified" not in r:
                r["attack_modified"] = modified
        if resp.get("reason") not in ("TEMPORAL_HIGH_FREQ", "TEMPORAL_STRAGGLER"):
            last_update_time[cid] = t_sub
        if resp.get("force_sync") is not None:
            last_update_time[cid] = resp["force_sync"].timestamp if hasattr(resp["force_sync"], "timestamp") else t_sub
        pulled_version[cid] = server.get_model_version()
        heapq.heappush(heap, (t_honest + float(period[cid] * rng.lognormal(0, time_sigma)), cid))
        step += 1

    return {"rows": logger.rows, "byz_ids": byz_ids, "server": server, "cfg": cfg}


# --------------------------------------------------------------------------------------
# Metrics (same quantities as the Kaggle comparison table)
# --------------------------------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]], warmup: int = 200) -> Dict[str, float]:
    """Same quantities as the Kaggle comparison table.

    ``*_w`` columns are the logged aggregation weight (includes the async staleness factor
    1/sqrt(1+version_lag)); ``*_dw`` columns divide that factor back out, i.e. the weight the
    DECISION ENGINE alone assigned (1.0 = full acceptance).
    """
    post = [r for r in rows if not r.get("is_warmup", False) and r.get("round", 0) >= warmup
            and r.get("reason") not in ("QUARANTINE_RELEASE_ACCEPT", "QUARANTINE_EXPIRED_REJECT")]
    out: Dict[str, float] = {}
    for tag, flag in (("honest", False), ("byz", True)):
        rs = [r for r in post if bool(r.get("is_byzantine")) == flag]
        n = max(1, len(rs))
        w = lambda r: float(r.get("weight") or 0.0)
        dw = lambda r: w(r) * float(np.sqrt(1.0 + (r.get("version_lag") or 0)))
        out[f"{tag}_n"] = len(rs)
        out[f"{tag}_full_accept_pct"] = 100.0 * sum(1 for r in rs if r["status"] == "ACCEPT") / n
        out[f"{tag}_downweight_pct"] = 100.0 * sum(1 for r in rs if r["status"] == "DOWNWEIGHT") / n
        out[f"{tag}_reject_pct"] = 100.0 * sum(1 for r in rs if r["status"] in ("REJECT", "QUARANTINE")) / n
        out[f"{tag}_admitted_pct"] = 100.0 * sum(1 for r in rs if r["status"] in ("ACCEPT", "DOWNWEIGHT")) / n
        out[f"{tag}_mean_w_all"] = float(np.sum([w(r) for r in rs]) / n)
        out[f"{tag}_mean_dw_all"] = float(np.sum([dw(r) for r in rs]) / n)
        adm = [dw(r) for r in rs if r["status"] in ("ACCEPT", "DOWNWEIGHT")]
        out[f"{tag}_mean_dw_admitted"] = float(np.mean(adm)) if adm else 0.0
    mods = [r for r in post if r.get("is_byzantine") and r.get("attack_modified")]
    nm = max(1, len(mods))
    wdw = lambda r: float(r.get("weight") or 0.0) * float(np.sqrt(1.0 + (r.get("version_lag") or 0)))
    out["byzmod_n"] = len(mods)
    out["byzmod_full_accept_pct"] = 100.0 * sum(1 for r in mods if r["status"] == "ACCEPT") / nm
    out["byzmod_admitted_pct"] = 100.0 * sum(1 for r in mods if r["status"] in ("ACCEPT", "DOWNWEIGHT")) / nm
    out["byzmod_mean_dw_all"] = float(np.sum([wdw(r) for r in mods]) / nm)
    hw, bw = out["honest_mean_dw_all"], out["byzmod_mean_dw_all"]
    out["honest_to_byzmod_dw_ratio"] = hw / bw if bw > 1e-9 else 999.0
    lags = [r.get("version_lag") or 0 for r in post if r["status"] == "ACCEPT"]
    out["mean_version_lag_accepted"] = float(np.mean(lags)) if lags else 0.0
    return out


def reason_table(rows, warmup=200, byz=False):
    post = [r for r in rows if not r.get("is_warmup", False) and r.get("round", 0) >= warmup
            and bool(r.get("is_byzantine")) == byz
            and r.get("reason") not in ("QUARANTINE_RELEASE_ACCEPT", "QUARANTINE_EXPIRED_REJECT")]
    cnt: Dict[str, int] = {}
    for r in post:
        k = f"{r['status']}:{r['reason']}"
        cnt[k] = cnt.get(k, 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


def evidence_stats(rows, warmup=200, byz=False):
    post = [r for r in rows if not r.get("is_warmup", False) and r.get("round", 0) >= warmup
            and bool(r.get("is_byzantine")) == byz]
    def col(k):
        v = [r.get(k) for r in post if r.get(k) is not None]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)
    return {k: col(k) for k in ("sim_global_raw", "sim_global", "trs_score", "subspace_mu",
                                "subspace_rho_parallel", "prc_score", "tra_score", "suspicion_score")}


def _parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = yaml.safe_load(v)
        except Exception:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="COMPOUND")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--D", type=int, default=3000)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--reasons", action="store_true")
    args = ap.parse_args()
    over = _parse_overrides(args.set)
    agg: Dict[str, List[float]] = {}
    for s in args.seeds:
        res = run_replay(over, attack=args.attack, seed=s, D=args.D, n_updates=args.updates)
        m = summarize(res["rows"])
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
        if args.reasons:
            print(f"seed {s} honest reasons:", reason_table(res["rows"], byz=False))
            print(f"seed {s} byz reasons:   ", reason_table(res["rows"], byz=True))
            print(f"seed {s} honest evidence:", evidence_stats(res["rows"], byz=False))
            print(f"seed {s} byz evidence:   ", evidence_stats(res["rows"], byz=True))
    print(f"attack={args.attack} seeds={args.seeds} overrides={over}")
    for k, v in agg.items():
        print(f"  {k:32s} mean={np.mean(v):9.3f}  (per-seed: {', '.join(f'{x:.2f}' for x in v)})")


if __name__ == "__main__":
    main()
