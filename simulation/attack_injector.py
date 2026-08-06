import torch
from typing import Optional, Tuple


MAX_BINARY_SEARCH_ITERS = 20

class AttackInjector:
    """Wraps a single Byzantine client. Given what an honest client would have
    submitted, returns the Byzantine-modified ``(delta_W, modified_g_i)``.

    Called by ``SimulationEnvironment`` during the training loop when a client
    is marked Byzantine.

    Supports six attack types:

    ====  =============  =====================================================
    Code  Name           Description
    ====  =============  =====================================================
    T1    HIGH_FREQ      Temporal: spams at 10x median rate, noise gradient
    T2    STRAGGLER      Temporal: submits 5x late, near-honest gradient
    S1    POISON         Spatial: sign-flipped -10x gradient, honest timing
    S2    MIMICRY        Spatial: epsilon-tuned to stay just above theta_cos
    --    ADAPTIVE       Combined: P_i-feedback timing + mild -1.5x poison
    --    COMPOUND       Combined: alternates T1 (even) / S2 (odd) per round
    ====  =============  =====================================================
    """

    def __init__(self, attack_type: str, client_id: int, config: dict) -> None:
        self.attack_type: str = attack_type
        self.client_id: int = client_id
        self.config: dict = config
        self._own_P_history: list[float] = []   # for ADAPTIVE attack
        self._round: int = 0                     # incremented on each inject() call
        self._dispatch = {
            "T1_HIGH_FREQ": self._t1_high_freq,
            "T2_STRAGGLER": self._t2_straggler,
            "S1_POISON":    self._s1_direct_poison,
            "S2_MIMICRY":   self._s2_mimicry,
            "ADAPTIVE":     self._adaptive_adversary,
            "COMPOUND":     self._compound,
        }

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def inject(
        self,
        honest_delta_W: torch.Tensor,
        context: dict,
    ) -> Tuple[torch.Tensor, float]:
        """Returns ``(modified_delta_W, modified_g_i)``.

        ``context`` dict keys (all provided by SimulationEnvironment):

        - ``"honest_g_i"`` (float): honest behavioral gap.
        - ``"median_g"`` (Optional[float]): median of server gap history.
        - ``"W_global"`` (torch.Tensor): current global model weights.
        - ``"ref_delta_W"`` (Optional[torch.Tensor]): top-K reference vector.
        - ``"theta_cos"`` (float): cosine similarity threshold.
        - ``"own_P_i"`` (float): this client's current pace score.
        """
        self._round += 1

        handler = self._dispatch.get(self.attack_type)
        if handler is None:
            raise ValueError(f"Unknown attack_type: {self.attack_type}")

        modified_dW, modified_g = handler(honest_delta_W, context)

        # Guard: catch NaN/Inf before they propagate through 500 rounds
        if not torch.isfinite(modified_dW).all():
            raise RuntimeError(
                f"AttackInjector [{self.attack_type}] produced non-finite "
                f"gradient at round {self._round} for client {self.client_id}"
            )

        return modified_dW, modified_g

    # ------------------------------------------------------------------
    # T1: High-frequency spammer
    # ------------------------------------------------------------------

    def _t1_high_freq(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Spammer: submits at ~10x median rate with random noise gradient.

        .. math::
            g' = 0.1 \cdot \text{median}(g)
            \quad
            \Delta W' = \frac{\mathbf{n}}{\|\mathbf{n}\|} \cdot 2\|\Delta W\|

        where :math:`\mathbf{n} \sim \mathcal{N}(0, I)`.
        """
        median_g = context.get("median_g")
        if median_g is not None:
            modified_g = median_g * 0.1
        else:
            modified_g = context["honest_g_i"] * 0.1

        noise = torch.randn_like(honest_delta_W)
        norm_noise = noise / (torch.norm(noise) + 1e-8)
        scale = torch.norm(honest_delta_W).item() * 2.0
        modified_dW = norm_noise * scale

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # T2: Strategic straggler
    # ------------------------------------------------------------------

    def _t2_straggler(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Strategic straggler: submits near-honest gradient very late.

        .. math::
            g' = 5.0 \cdot \text{median}(g)
            \quad
            \Delta W' = \Delta W + 0.05 \cdot \mathbf{n}

        Falls back to :math:`5 \cdot g_{\text{honest}}` if median is
        unavailable (early rounds).
        """
        median_g = context.get("median_g")
        if median_g is not None:
            modified_g = median_g * 5.0
        else:
            modified_g = context["honest_g_i"] * 5.0

        stale_noise = torch.randn_like(honest_delta_W) * 0.05
        modified_dW = honest_delta_W + stale_noise

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # S1: Direct poisoner
    # ------------------------------------------------------------------

    def _s1_direct_poison(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Direct poisoner: sign-flipped, scaled gradient with honest timing.

        .. math::
            \Delta W' = -10 \cdot \Delta W
            \quad
            g' = g_{\text{honest}}

        Note: L2 clipping will reduce the inflated norm; the sign-flip
        is the real attack vector.
        """
        modified_dW = -10.0 * honest_delta_W
        modified_g = context["honest_g_i"]
        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # S2: Mimicry poisoner
    # ------------------------------------------------------------------

    def _s2_mimicry(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Mimicry poisoner: maximum perturbation that stays above
        :math:`\theta_{\cos} + 0.05`.

        Generates an orthogonal perturbation direction via Gram-Schmidt,
        then binary-searches for the largest :math:`\varepsilon` such that:

        .. math::
            \cos(\Delta W + \varepsilon \mathbf{v},\; \mathbf{r})
            \;\geq\; \theta_{\cos} + 0.05

        Assumes white-box access to the reference vector and threshold
        (standard worst-case adversary assumption in Byzantine FL).
        """
        modified_g = context["honest_g_i"]

        ref_dW: Optional[torch.Tensor] = context.get("ref_delta_W")
        if ref_dW is None:
            return (honest_delta_W, modified_g)

        ref = ref_dW.flatten()
        if torch.norm(ref) < 1e-8:
            return (honest_delta_W, modified_g)

        # Generate perturbation direction orthogonal to ref (Gram-Schmidt)
        v = torch.randn_like(ref)
        v = v - (torch.dot(v, ref) / (torch.dot(ref, ref) + 1e-8)) * ref
        v = v / (torch.norm(v) + 1e-8)

        # Binary search for maximum epsilon that still passes cosine check
        theta_target = context["theta_cos"] + 0.05
        lo, hi = 0.0, torch.norm(honest_delta_W).item() * 2
        for _ in range(MAX_BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2
            candidate = honest_delta_W.flatten() + mid * v
            sim = torch.dot(candidate, ref) / (
                torch.norm(candidate) * torch.norm(ref) + 1e-8
            )
            if sim.item() >= theta_target:
                lo = mid    # can perturb more and still pass
            else:
                hi = mid    # reduce perturbation to pass
        epsilon = lo

        modified_dW = (
            honest_delta_W.flatten() + epsilon * v
        ).reshape(honest_delta_W.shape)

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # Adaptive adversary
    # ------------------------------------------------------------------

    def _adaptive_adversary(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Adaptive adversary: modulates timing based on P_i trend.

        Timing rule (thresholds are attack-model constants, not config):

        - :math:`\Delta P < -0.05` (falling): speed up to :math:`0.7 \cdot g`
        - :math:`\Delta P > +0.05` (recovering): slow to :math:`1.1 \cdot g`
        - otherwise: maintain :math:`g_{\text{honest}}`

        Gradient: :math:`\Delta W' = -1.5 \cdot \Delta W` (mild poison).

        On the first round (< 2 P_i observations), defaults to T1 behavior.
        """
        self._own_P_history.append(context["own_P_i"])

        # Fewer than 2 observations -- behave as T1
        if len(self._own_P_history) < 2:
            return self._t1_high_freq(honest_delta_W, context)

        delta_P = self._own_P_history[-1] - self._own_P_history[-2]

        if delta_P < -0.05:
            # P_i falling -- speed up
            modified_g = context["honest_g_i"] * 0.7
        elif delta_P > 0.05:
            # P_i recovering -- slowly probe upper fence
            modified_g = context["honest_g_i"] * 1.1
        else:
            modified_g = context["honest_g_i"]

        modified_dW = -1.5 * honest_delta_W
        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # Compound attack
    # ------------------------------------------------------------------

    def _compound(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        """Compound: alternates between T1 spam and S2 mimicry.

        Note: ``self._round`` is incremented before dispatch, so round 1
        is odd (S2 mimicry) and round 2 is even (T1 high-freq).
        """
        if self._round % 2 == 0:
            return self._t1_high_freq(honest_delta_W, context)
        else:
            return self._s2_mimicry(honest_delta_W, context)
