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
        
        # Write headers if file doesn't exist or is empty
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["round", "client_id", "status", "reason", "g_i", "I_i", "P_i"])

    def log_update(self, *, round: int, client_id: int, status: str, reason: str, 
                   g_i: Optional[float] = None, I_i: Optional[float] = None, P_i: Optional[float] = None) -> None:
        """Log update status and metadata. Appends to list and CSV."""
        entry = {
            "round": round,
            "client_id": client_id,
            "status": status,
            "reason": reason,
            "g_i": g_i,
            "I_i": I_i,
            "P_i": P_i
        }
        self._rejection_log.append(entry)
        
        # Format values for CSV
        g_val = f"{g_i:.6f}" if g_i is not None else ""
        I_val = f"{I_i:.6f}" if I_i is not None else ""
        P_val = f"{P_i:.6f}" if P_i is not None else ""
        
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([round, client_id, status, reason, g_val, I_val, P_val])

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
