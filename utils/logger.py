import os
import csv
from typing import List, Dict, Any, Optional

class BDSFLogger:
    def __init__(self, run_id: str, config: dict):
        self.run_id = run_id
        self._rejection_log: List[Dict[str, Any]] = []
        self._reputation_log: List[Dict[str, Any]] = []
        self._metric_log: List[Dict[str, Any]] = []
        
        # Get log directory and ensure it exists
        self.log_dir = config.get("log_dir", "logs/")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup CSV file for updates
        self.csv_path = os.path.join(self.log_dir, f"{run_id}_updates.csv")
        headers = [
            "round", "client_id", "status", "reason",
            "g_i", "version_lag", "I_i", "P_i",
            "lower_fence", "upper_fence", "fence_margin", "client_z_score", "is_burn_in",
            "spatial_mature", "temporal_mature", "behavioral_mature", "spatial_ref_count", "spatial_coherence",
            "sim_global", "norm_raw", "norm_clipped", "norm_ratio_median", "dynamic_bound_C", "reference_available",
            "weight", "action",
            "sim_self_mean", "sim_self_max", "norm_deviation_self", "cadence_consistency", "history_depth",
            "sim_anchor", "sim_frozen_anchor", "anchor_drift", "consecutive_dw", "quarantine_depth",
            "v_momentum_norm"
        ]
        
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
                writer.writerow(headers)
                for r in clean_rows:
                    writer.writerow(r)
                f.flush()
        else:
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                f.flush()

    def log_update(self, *, round: int, client_id: int, status: str, reason: str, 
                   g_i: Optional[float] = None, version_lag: Optional[int] = None,
                   I_i: Optional[float] = None, P_i: Optional[float] = None,
                   lower_fence: Optional[float] = None, upper_fence: Optional[float] = None,
                   fence_margin: Optional[float] = None, client_z_score: Optional[float] = None,
                   is_burn_in: Optional[bool] = None,
                   spatial_mature: Optional[bool] = None, temporal_mature: Optional[bool] = None,
                   behavioral_mature: Optional[bool] = None, spatial_ref_count: Optional[int] = None,
                   spatial_coherence: Optional[float] = None,
                   sim_global: Optional[float] = None,
                   norm_raw: Optional[float] = None, norm_clipped: Optional[float] = None,
                   norm_ratio_median: Optional[float] = None, dynamic_bound_C: Optional[float] = None,
                   reference_available: Optional[bool] = None, weight: Optional[float] = None,
                   action: Optional[str] = None,
                   sim_self_mean: Optional[float] = None, sim_self_max: Optional[float] = None,
                   norm_deviation_self: Optional[float] = None, cadence_consistency: Optional[float] = None,
                   history_depth: Optional[int] = None,
                   sim_anchor: Optional[float] = None,
                   sim_frozen_anchor: Optional[float] = None,
                   anchor_drift: Optional[float] = None,
                   consecutive_dw: Optional[int] = None,
                   quarantine_depth: Optional[int] = None,
                   v_momentum_norm: Optional[float] = None) -> None:
        """Log update status and metadata. Appends to list and CSV."""
        act_val = action if action is not None else status
        entry = {
            "round": round,
            "client_id": client_id,
            "status": status,
            "reason": reason,
            "g_i": g_i,
            "version_lag": version_lag,
            "I_i": I_i,
            "P_i": P_i,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "fence_margin": fence_margin,
            "client_z_score": client_z_score,
            "is_burn_in": is_burn_in,
            "spatial_mature": spatial_mature,
            "temporal_mature": temporal_mature,
            "behavioral_mature": behavioral_mature,
            "spatial_ref_count": spatial_ref_count,
            "spatial_coherence": spatial_coherence,
            "sim_global": sim_global,
            "norm_raw": norm_raw,
            "norm_clipped": norm_clipped,
            "norm_ratio_median": norm_ratio_median,
            "dynamic_bound_C": dynamic_bound_C,
            "reference_available": reference_available,
            "weight": weight,
            "action": act_val,
            "sim_self_mean": sim_self_mean,
            "sim_self_max": sim_self_max,
            "norm_deviation_self": norm_deviation_self,
            "cadence_consistency": cadence_consistency,
            "history_depth": history_depth,
            "sim_anchor": sim_anchor,
            "sim_frozen_anchor": sim_frozen_anchor,
            "anchor_drift": anchor_drift,
            "consecutive_dw": consecutive_dw,
            "quarantine_depth": quarantine_depth,
            "v_momentum_norm": v_momentum_norm,
        }
        self._rejection_log.append(entry)
        
        # Format values for CSV
        g_val = f"{g_i:.6f}" if g_i is not None else ""
        vlag_val = str(version_lag) if version_lag is not None else ""
        I_val = f"{I_i:.6f}" if I_i is not None else ""
        P_val = f"{P_i:.6f}" if P_i is not None else ""
        lf_val = f"{lower_fence:.6f}" if lower_fence is not None else ""
        uf_val = f"{upper_fence:.6f}" if upper_fence is not None else ""
        fm_val = f"{fence_margin:.6f}" if fence_margin is not None else ""
        cz_val = f"{client_z_score:.6f}" if client_z_score is not None else ""
        bi_val = str(is_burn_in) if is_burn_in is not None else ""
        sp_m_val = str(spatial_mature) if spatial_mature is not None else ""
        tm_m_val = str(temporal_mature) if temporal_mature is not None else ""
        bm_m_val = str(behavioral_mature) if behavioral_mature is not None else ""
        src_val = str(spatial_ref_count) if spatial_ref_count is not None else ""
        sc_val = f"{spatial_coherence:.6f}" if spatial_coherence is not None else ""
        sim_g_val = f"{sim_global:.6f}" if sim_global is not None else ""
        nr_val = f"{norm_raw:.6f}" if norm_raw is not None else ""
        nc_val = f"{norm_clipped:.6f}" if norm_clipped is not None else ""
        nrm_val = f"{norm_ratio_median:.6f}" if norm_ratio_median is not None else ""
        dbc_val = f"{dynamic_bound_C:.6f}" if dynamic_bound_C is not None else ""
        ra_val = str(reference_available) if reference_available is not None else ""
        w_val = f"{weight:.6f}" if weight is not None else ""
        ss_mean_val = f"{sim_self_mean:.6f}" if sim_self_mean is not None else ""
        ss_max_val = f"{sim_self_max:.6f}" if sim_self_max is not None else ""
        nd_self_val = f"{norm_deviation_self:.6f}" if norm_deviation_self is not None else ""
        cc_val = f"{cadence_consistency:.6f}" if cadence_consistency is not None else ""
        hd_val = str(history_depth) if history_depth is not None else ""
        sa_val = f"{sim_anchor:.6f}" if sim_anchor is not None else ""
        sfa_val = f"{sim_frozen_anchor:.6f}" if sim_frozen_anchor is not None else ""
        adr_val = f"{anchor_drift:.6f}" if anchor_drift is not None else ""
        cdw_val = str(consecutive_dw) if consecutive_dw is not None else ""
        qd_val = str(quarantine_depth) if quarantine_depth is not None else ""
        vm_val = f"{v_momentum_norm:.6f}" if v_momentum_norm is not None else ""
        
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round, client_id, status, reason,
                g_val, vlag_val, I_val, P_val,
                lf_val, uf_val, fm_val, cz_val, bi_val,
                sp_m_val, tm_m_val, bm_m_val, src_val, sc_val,
                sim_g_val, nr_val, nc_val, nrm_val, dbc_val, ra_val,
                w_val, act_val,
                ss_mean_val, ss_max_val, nd_self_val, cc_val, hd_val,
                sa_val, sfa_val, adr_val, cdw_val, qd_val,
                vm_val
            ])
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
        headers = [
            "round", "client_id", "status", "reason",
            "g_i", "version_lag", "I_i", "P_i",
            "lower_fence", "upper_fence", "fence_margin", "client_z_score", "is_burn_in",
            "spatial_mature", "temporal_mature", "behavioral_mature", "spatial_ref_count", "spatial_coherence",
            "sim_global", "norm_raw", "norm_clipped", "norm_ratio_median", "dynamic_bound_C", "reference_available",
            "weight", "action",
            "sim_self_mean", "sim_self_max", "norm_deviation_self", "cadence_consistency", "history_depth",
            "sim_anchor", "sim_frozen_anchor", "anchor_drift", "consecutive_dw", "quarantine_depth",
            "v_momentum_norm"
        ]
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
            writer.writerow(headers)
            for r in clean_rows:
                writer.writerow(r)
            f.flush()
        
        self._rejection_log = [e for e in self._rejection_log if e.get("round", 0) < resume_update]
