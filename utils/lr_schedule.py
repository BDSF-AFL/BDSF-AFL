"""Single source of truth for the round value fed to LocalTrainer's learning-rate schedule."""


def lr_schedule_round(server) -> float:
    """Round index (float) for the lr schedule.

    ``server.round_number`` counts UPDATES, while the schedule's ``total_rounds`` is in ROUNDS (one round =
    N_clients updates). Honest clients (ClientNode.run_one_round) and Byzantine clients
    (AttackInjectorWrapper.run_one_round) must both use this, otherwise one group decays N times faster.
    ``lr_schedule_unit: "update"`` restores the legacy behaviour.
    """
    updates = getattr(server, "round_number", 0)
    cfg = getattr(server, "config", None)
    if isinstance(cfg, dict) and str(cfg.get("lr_schedule_unit", "round")) == "round":
        return updates / max(1, int(cfg.get("N_clients", 20)))
    return updates
