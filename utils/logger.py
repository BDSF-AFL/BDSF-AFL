import os
import csv
from typing import List, Dict, Any, Optional
import numpy as np

# Canonical 44-column CSV Logging Schema for BDSF-AFL v2 Architecture
BDSF_UPDATE_HEADERS: List[str] = [
    # Core update metadata & ground truth
    "round",
    "client_id",
    "is_byzantine",
    "status",
    "reason",
    "weight",
    "priority",
    "is_warmup",

    # Pillar 2: Reputation
    "I_i",
    "P_i",

    # Pillar 1: Temporal Cadence & Delay
    "g_i",
    "version_lag",
    "lower_fence",
    "upper_fence",
    "fence_margin",
    "temporal_mature",

    # Pillar 3: Spatial Reference & Adaptive Norms
    "sim_global",
    "norm_raw",
    "norm_clipped",
    "norm_ratio_median",
    "dynamic_bound_C",
    "spatial_coherence",
    "spatial_mature",

    # Pillar 4: Dual-Anchor & Behavioral Memory
    "sim_self_max",
    "sim_anchor",
    "sim_frozen_anchor",
    "anchor_drift",
    "history_depth",
    "behavioral_mature",

    # Pillar 5: Residual Coherence, Rigidity & Suspicion
    "prc_score",
    "tra_score",
    "suspicion_score",
    "gdv_score",
    "dbp_score",
    "trs_score",

    # Subspace Projection Engine (BDSF-AFL v2 / Gates 1, 1b, 2)
    "subspace_c_min",
    "subspace_c_sum",
    "subspace_norm_perp",
    "subspace_M_perp",
    "subspace_w_damp",
    "subspace_w_temporal",
    "subspace_rho",
    "subspace_basis_count",

    # Server Velocity
    "v_momentum_norm",
]


class BDSFLogger:
    """Comprehensive structured logger for BDSF-AFL experiments.
    
    Maintains in-memory diagnostic logs and appends structured rows to a canonical
    per-run CSV file following the full 44-column modern architecture schema.
    """

    def __init__(self, run_id: str, config: dict):
        self.run_id = run_id
        self._rejection_log: List[Dict[str, Any]] = []
        self._reputation_log: List[Dict[str, Any]] = []
        self._metric_log: List[Dict[str, Any]] = []
        self.headers = list(BDSF_UPDATE_HEADERS)
        
        # Get log directory and ensure it exists
        self.log_dir = config.get("log_dir", "logs/")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup CSV file for updates
        self.csv_path = os.path.join(self.log_dir, f"{run_id}_updates.csv")
        
        is_resume = config.get("resume", False) and os.path.exists(self.csv_path)
        resume_round = config.get("resume_round", None)
        
        if is_resume:
            clean_rows = []
            try:
                with open(self.csv_path, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    for r in reader:
                        if r and len(r) > 0:
                            if resume_round is not None:
                                try:
                                    r_num = int(r[0])
                                    if r_num <= resume_round:
                                        clean_rows.append(r)
                                except ValueError:
                                    clean_rows.append(r)
                            else:
                                clean_rows.append(r)
            except Exception:
                clean_rows = []
            
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                for r in clean_rows:
                    writer.writerow(r)
                f.flush()
        else:
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                f.flush()

    @staticmethod
    def _format_cell(val: Any) -> str:
        """Formats primitive, numeric, and boolean values for consistent CSV serialization."""
        if val is None:
            return ""
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return ""
            return f"{val:.6f}"
        return str(val)

    def log_update(
        self,
        *,
        round: int,
        client_id: int,
        status: str,
        reason: str,
        is_byzantine: Optional[bool] = None,
        weight: Optional[float] = None,
        priority: Optional[int] = None,
        is_warmup: Optional[bool] = None,
        I_i: Optional[float] = None,
        P_i: Optional[float] = None,
        g_i: Optional[float] = None,
        version_lag: Optional[int] = None,
        lower_fence: Optional[float] = None,
        upper_fence: Optional[float] = None,
        fence_margin: Optional[float] = None,
        temporal_mature: Optional[bool] = None,
        sim_global: Optional[float] = None,
        norm_raw: Optional[float] = None,
        norm_clipped: Optional[float] = None,
        norm_ratio_median: Optional[float] = None,
        dynamic_bound_C: Optional[float] = None,
        spatial_coherence: Optional[float] = None,
        spatial_mature: Optional[bool] = None,
        sim_self_max: Optional[float] = None,
        sim_anchor: Optional[float] = None,
        sim_frozen_anchor: Optional[float] = None,
        anchor_drift: Optional[float] = None,
        history_depth: Optional[int] = None,
        behavioral_mature: Optional[bool] = None,
        prc_score: Optional[float] = None,
        tra_score: Optional[float] = None,
        suspicion_score: Optional[float] = None,
        gdv_score: Optional[float] = None,
        dbp_score: Optional[float] = None,
        trs_score: Optional[float] = None,
        subspace_c_min: Optional[float] = None,
        subspace_c_sum: Optional[float] = None,
        subspace_norm_perp: Optional[float] = None,
        subspace_M_perp: Optional[float] = None,
        subspace_w_damp: Optional[float] = None,
        subspace_w_temporal: Optional[float] = None,
        subspace_rho: Optional[float] = None,
        subspace_basis_count: Optional[int] = None,
        v_momentum_norm: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log client update status and diagnostic metadata across all architecture pillars.
        
        Appends a structured dictionary entry to the in-memory rejection log and writes
        a formatted row to the run CSV file.
        """
        entry: Dict[str, Any] = {
            "round": round,
            "client_id": client_id,
            "is_byzantine": is_byzantine,
            "status": status,
            "reason": reason,
            "weight": weight,
            "priority": priority,
            "is_warmup": is_warmup,
            "I_i": I_i,
            "P_i": P_i,
            "g_i": g_i,
            "version_lag": version_lag,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "fence_margin": fence_margin,
            "temporal_mature": temporal_mature,
            "sim_global": sim_global,
            "norm_raw": norm_raw,
            "norm_clipped": norm_clipped,
            "norm_ratio_median": norm_ratio_median,
            "dynamic_bound_C": dynamic_bound_C,
            "spatial_coherence": spatial_coherence,
            "spatial_mature": spatial_mature,
            "sim_self_max": sim_self_max,
            "sim_anchor": sim_anchor,
            "sim_frozen_anchor": sim_frozen_anchor,
            "anchor_drift": anchor_drift,
            "history_depth": history_depth,
            "behavioral_mature": behavioral_mature,
            "prc_score": prc_score,
            "tra_score": tra_score,
            "suspicion_score": suspicion_score,
            "gdv_score": gdv_score,
            "dbp_score": dbp_score,
            "trs_score": trs_score,
            "subspace_c_min": subspace_c_min,
            "subspace_c_sum": subspace_c_sum,
            "subspace_norm_perp": subspace_norm_perp,
            "subspace_M_perp": subspace_M_perp,
            "subspace_w_damp": subspace_w_damp,
            "subspace_w_temporal": subspace_w_temporal,
            "subspace_rho": subspace_rho,
            "subspace_basis_count": subspace_basis_count,
            "v_momentum_norm": v_momentum_norm,
        }
        # Absorb any remaining kwargs matching valid headers
        for k, v in kwargs.items():
            if k in self.headers and entry.get(k) is None:
                entry[k] = v

        self._rejection_log.append(entry)

        # Write to disk
        row = [self._format_cell(entry.get(h)) for h in self.headers]
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()

    def log_reputation(self, *, round: int, client_id: int, I_i: float, P_i: float, is_byzantine: bool) -> None:
        """Log client reputation metrics at a round."""
        self._reputation_log.append({
            "round": round,
            "client_id": client_id,
            "I_i": I_i,
            "P_i": P_i,
            "is_byzantine": is_byzantine
        })

    def log_metric(self, *, round: int, metric_name: str, value: float) -> None:
        """Log general evaluation metrics (e.g. accuracy)."""
        self._metric_log.append({
            "round": round,
            "metric_name": metric_name,
            "value": value
        })

    def get_rejection_log(self) -> List[Dict[str, Any]]:
        return self._rejection_log

    def get_reputation_log(self) -> List[Dict[str, Any]]:
        return self._reputation_log

    def get_metric_log(self) -> List[Dict[str, Any]]:
        return self._metric_log

    def truncate_csv_at_round(self, resume_update: int) -> None:
        """Truncates the CSV log to retain only rows recorded strictly before resume_update, eliminating duplicates on resume."""
        if not os.path.exists(self.csv_path):
            return

        clean_rows = []
        try:
            with open(self.csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for r in reader:
                    if r and len(r) > 0:
                        try:
                            r_num = int(r[0])
                            if r_num < resume_update:
                                clean_rows.append(r)
                        except ValueError:
                            pass
        except Exception:
            clean_rows = []

        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for r in clean_rows:
                writer.writerow(r)
            f.flush()
        
        self._rejection_log = [e for e in self._rejection_log if e.get("round", 0) < resume_update]
