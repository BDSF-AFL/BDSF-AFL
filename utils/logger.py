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
            "round", "client_id", "status", "reason", "weight",
            "I_i", "P_i",
            "g_i", "version_lag", "lower_fence", "upper_fence", "fence_margin", "temporal_mature",
            "sim_global", "norm_raw", "norm_ratio_median", "spatial_coherence", "spatial_mature",
            "sim_self_max", "sim_anchor", "behavioral_mature",
            "prc_score", "tra_score", "suspicion_score",
            "gdv_score", "dbp_score", "trs_score"
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

    def log_update(
        self,
        *,
        round: int,
        client_id: int,
        status: str,
        reason: str,
        weight: Optional[float] = None,
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
        norm_ratio_median: Optional[float] = None,
        spatial_coherence: Optional[float] = None,
        spatial_mature: Optional[bool] = None,
        sim_self_max: Optional[float] = None,
        sim_anchor: Optional[float] = None,
        behavioral_mature: Optional[bool] = None,
        prc_score: Optional[float] = None,
        tra_score: Optional[float] = None,
        suspicion_score: Optional[float] = None,
        gdv_score: Optional[float] = None,
        dbp_score: Optional[float] = None,
        trs_score: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log update status and metadata. Appends to list and CSV."""
        entry = {
            "round": round,
            "client_id": client_id,
            "status": status,
            "reason": reason,
            "weight": weight,
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
            "norm_ratio_median": norm_ratio_median,
            "spatial_coherence": spatial_coherence,
            "spatial_mature": spatial_mature,
            "sim_self_max": sim_self_max,
            "sim_anchor": sim_anchor,
            "behavioral_mature": behavioral_mature,
            "prc_score": prc_score,
            "tra_score": tra_score,
            "suspicion_score": suspicion_score,
            "gdv_score": gdv_score,
            "dbp_score": dbp_score,
            "trs_score": trs_score,
        }
        self._rejection_log.append(entry)
        
        # Format values for CSV
        w_val = f"{weight:.6f}" if weight is not None else ""
        I_val = f"{I_i:.6f}" if I_i is not None else ""
        P_val = f"{P_i:.6f}" if P_i is not None else ""
        g_val = f"{g_i:.6f}" if g_i is not None else ""
        vlag_val = str(version_lag) if version_lag is not None else ""
        lf_val = f"{lower_fence:.6f}" if lower_fence is not None else ""
        uf_val = f"{upper_fence:.6f}" if upper_fence is not None else ""
        fm_val = f"{fence_margin:.6f}" if fence_margin is not None else ""
        tm_m_val = str(temporal_mature) if temporal_mature is not None else ""
        sim_g_val = f"{sim_global:.6f}" if sim_global is not None else ""
        nr_val = f"{norm_raw:.6f}" if norm_raw is not None else ""
        nrm_val = f"{norm_ratio_median:.6f}" if norm_ratio_median is not None else ""
        sc_val = f"{spatial_coherence:.6f}" if spatial_coherence is not None else ""
        sp_m_val = str(spatial_mature) if spatial_mature is not None else ""
        ss_max_val = f"{sim_self_max:.6f}" if sim_self_max is not None else ""
        sa_val = f"{sim_anchor:.6f}" if sim_anchor is not None else ""
        bm_m_val = str(behavioral_mature) if behavioral_mature is not None else ""
        prc_val = f"{prc_score:.6f}" if prc_score is not None else ""
        tra_val = f"{tra_score:.6f}" if tra_score is not None else ""
        susp_val = f"{suspicion_score:.6f}" if suspicion_score is not None else ""
        gdv_val = f"{gdv_score:.6f}" if gdv_score is not None else ""
        dbp_val = f"{dbp_score:.6f}" if dbp_score is not None else ""
        trs_val = f"{trs_score:.6f}" if trs_score is not None else ""
        
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round, client_id, status, reason, w_val,
                I_val, P_val,
                g_val, vlag_val, lf_val, uf_val, fm_val, tm_m_val,
                sim_g_val, nr_val, nrm_val, sc_val, sp_m_val,
                ss_max_val, sa_val, bm_m_val,
                prc_val, tra_val, susp_val,
                gdv_val, dbp_val, trs_val
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
            "round", "client_id", "status", "reason", "weight",
            "I_i", "P_i",
            "g_i", "version_lag", "lower_fence", "upper_fence", "fence_margin", "temporal_mature",
            "sim_global", "norm_raw", "norm_ratio_median", "spatial_coherence", "spatial_mature",
            "sim_self_max", "sim_anchor", "behavioral_mature",
            "prc_score", "tra_score", "suspicion_score",
            "gdv_score", "dbp_score", "trs_score"
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
