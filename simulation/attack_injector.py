import torch
from typing import Optional, Tuple


MAX_BINARY_SEARCH_ITERS = 20

class AttackInjector:
    """Wraps a single Byzantine client. Given what an honest client would have
    submitted, returns the Byzantine-modified ``(delta_W, modified_g_i)``.

    Called by ``SimulationEnvironment`` during the (asynchronous) training
    loop when a client is marked Byzantine.

    Supports six attack types:

    ====  =============  =====================================================
    Code  Name           Description
    ====  =============  =====================================================
    T1    HIGH_FREQ      Temporal (async): floods at ~10x median submission
                          rate, Gaussian/random-noise payload
    T2    STRAGGLER      Temporal (async): submits ~5x late, near-honest
                          gradient plus small additive noise
    S1    POISON         Spatial: sign-flipped, scaled gradient, honest timing
    S2    MIMICRY        Spatial: directed-deviation perturbation -- maximal
                          push against the trust anchor while still clearing
                          the cosine floor  [logic fixed, see changelog]
    --    ADAPTIVE       Combined: P_i-feedback timing + directed-deviation
                          spatial payload with bounded morphing jitter
                          [logic fixed, see changelog]
    --    COMPOUND       Combined: alternates T1 (even) / S2 (odd) per round
    ====  =============  =====================================================

    Research grounding
    -------------------
    - T1 (payload):    Gaussian / random attack -- Blanchard, El Mhamdi,
                        Guerraoui & Stainer, "Machine Learning with
                        Adversaries: Byzantine Tolerant Gradient Descent",
                        NeurIPS 2017 (the Krum paper's attack baseline).
    - T1 (threat class): high-submission-frequency Byzantine workers are the
                        explicit threat Kardam's frequency filter targets --
                        Damaskinos, El Mhamdi, Guerraoui, Patra & Taziki,
                        "Asynchronous Byzantine Machine Learning (the case
                        of SGD)", ICML 2018.
    - T2 (payload):    disguised free-rider additive-noise construction
                        (near-honest update + small noise to look
                        plausible without doing real work) -- Fraboni,
                        Vidal & Lorenzi, "Free-rider Attacks on Model
                        Aggregation in Federated Learning", AISTATS 2021.
    - T2 (threat class): staleness/delay exploitation in async FL is
                        surveyed in Damaskinos et al. 2018 (above) and
                        Cox, Malan, Chen & Decouchant, "Asynchronous
                        Byzantine Federated Learning", arXiv 2024.
    - S1:              Sign-Flipping (SF) -- Allen-Zhu, Ebrahimianghazani,
                        Li & Alistarh, "Byzantine-Resilient Non-Convex
                        Stochastic Gradient Descent", ICLR 2020. The -10x
                        scale already matches the exact convention used
                        for SF under worker delays in Xie, Koyejo & Gupta,
                        "Zeno++: Robust Fully Asynchronous SGD", ICML 2020
                        (their Fig. 4-5 caption: "g will be replaced by
                        -10g").
    - S2 / ADAPTIVE:   cosine-constrained instance of the generic
                        optimization-poisoning framework -- Shejwalkar &
                        Houmansadr, "Manipulating the Byzantine: Optimizing
                        Model Poisoning Attacks and Defenses for FL",
                        NDSS 2021 (Min-Max/Min-Sum; we reuse their
                        empirically-strongest "inverse unit vector"
                        perturbation direction), and Fang, Cao, Jia & Gong,
                        "Local Model Poisoning Attacks to Byzantine-Robust
                        FL", USENIX Security 2020 (defense-aware
                        optimization attacks in general). Evaluating
                        defenses against an attacker with knowledge of the
                        defense mechanism (ADAPTIVE) is the methodology
                        argued for in Shejwalkar, Houmansadr, Kairouz &
                        Ramage, "Back to the Drawing Board: A Critical
                        Evaluation of Poisoning Attacks on Production FL",
                        IEEE S&P 2022.

    A note on the name "S2_MIMICRY": the FL literature has a specific,
    different "Mimic" attack (Karimireddy, He & Jaggi, ICLR 2022) where
    Byzantine workers replicate a single honest worker's update to bias
    aggregation on heterogeneous data. That is NOT what this function
    does -- it runs a directed-deviation / Min-Max-style optimization
    attack against the cosine-similarity spatial check. The dispatch key
    is left unchanged since renaming it would ripple through the rest of
    the codebase, but it is worth knowing the name is a false friend if a
    reviewer familiar with Karimireddy et al. reads the paper.

    Fix changelog (this revision)
    -------------------------------
    - S2_MIMICRY [critical, logic bug]: the previous perturbation
      direction was constructed *orthogonal* to the reference vector via
      Gram-Schmidt from a random per-client seed. Because the direction
      was orthogonal, cos(candidate, ref) was NOT monotonic in the search
      variable epsilon -- it rises, peaks, then falls -- so the bisection
      (which assumes monotonic decrease) converged to an arbitrary point
      rather than the true maximum whenever the honest gradient had a
      negative component along that random direction (roughly half of all
      draws). This silently produced a much weaker attack than intended
      about half the time, with no error raised. Fixed by switching to
      the literature-standard "inverse unit vector" direction
      v = -ref/||ref|| (Shejwalkar & Houmansadr, 2021), for which
      monotonicity is provable by construction (see
      `_directed_epsilon_max`'s docstring) -- the bisection is now sound
      for every draw, and the attack is also strictly stronger, since
      pushing directly against the trust anchor is more damaging than an
      orthogonal offset. The static seeded "malicious objective" is no
      longer needed and has been removed.
      Empirical check (d=200, 400 random trials, theta_target=0.55):
      the old orthogonal direction produced a genuinely non-monotonic
      cosine curve in 47.0% of draws. Most of those cost only a little
      accuracy in high dimension -- concentration of measure keeps a
      random orthogonal component's contribution small relative to
      ||delta_W||, so the bisection lands close to the true optimum
      anyway -- but 0.5% of draws were unambiguous failures: the true
      feasible window excluded epsilon=0 entirely, and the old code
      (which implicitly assumes epsilon=0 is always a valid starting
      point) returned ~0 instead of the real, much larger optimum that
      existed further out. Which draws are affected is unpredictable
      per client per round, so this quietly added unquantified
      noise/bias into any attack-strength numbers computed from the old
      code, and the failure rate is expected to be *higher* for
      lower-dimensional per-layer tensors or for theta_cos values tuned
      close to where honest updates naturally sit -- i.e. exactly the
      regime a well-tuned defense's threshold usually lives in.
    - S2_MIMICRY [robustness]: the previous upper search bound
      (2 * ||delta_W||) was a fixed guess that is not guaranteed to
      bracket the true root for all values of theta_cos. Replaced with an
      exponential bracket-expansion step before bisecting, and an
      explicit epsilon=0 feasibility check (if even the honest update
      can't clear the margin, it is returned unmodified rather than the
      old code's undefined behaviour).
    - ADAPTIVE [same critical bug]: the spatial payload reused the same
      orthogonal Gram-Schmidt construction as the old S2 and inherited
      the identical monotonicity bug. Fixed the same way, with a small
      verified-and-bounded orthogonal jitter layered on top (not a raw
      unchecked one) so the round-to-round "morphing" behaviour survives
      without reopening the soundness problem.
    - T1 / T2 / S1: no logic bugs found. Left mathematically as-is;
      hardcoded magic numbers (noise scale, sign-flip scale) were pulled
      into `self.config` with the original values as defaults, so
      existing experiment results are reproducible unless you opt into
      different constants. Docstrings now cite the papers each already
      matches.
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
        r"""Async flooder: submits at ~10x median rate carrying a
        Gaussian/random-attack payload.

        Payload matches the canonical random attack of Blanchard et al.
        (NeurIPS 2017): a random direction scaled to a multiple of the
        honest update's own norm. The submission-frequency behavior this
        models -- a Byzantine worker flooding an async aggregator to
        drown out honest, slower workers -- is exactly the threat Kardam's
        frequency filter defends against (Damaskinos et al., ICML 2018).
        No logic bug found here; the noise-norm multiplier is now a
        config knob defaulting to the original hardcoded value.

        .. math::
            g' = 0.1 \cdot \text{median}(g)
            \quad
            \Delta W' = \frac{\mathbf{n}}{\|\mathbf{n}\|} \cdot k\|\Delta W\|

        where :math:`\mathbf{n} \sim \mathcal{N}(0, I)` and :math:`k`
        defaults to 2 (``config["t1_noise_norm_mult"]``).
        """
        median_g = context.get("median_g")
        if median_g is not None:
            modified_g = median_g * 0.1
        else:
            modified_g = context["honest_g_i"] * 0.1

        noise = torch.randn_like(honest_delta_W)
        norm_noise = noise / (torch.norm(noise) + 1e-8)
        scale_mult = self.config.get("t1_noise_norm_mult", 2.0)
        scale = torch.norm(honest_delta_W).item() * scale_mult
        modified_dW = norm_noise * scale

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # T2: Strategic straggler
    # ------------------------------------------------------------------

    def _t2_straggler(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Strategic straggler: submits a near-honest gradient very late.

        Payload structure (honest update + small additive noise) mirrors
        the "disguised free-rider" construction of Fraboni, Vidal &
        Lorenzi (AISTATS 2021): stay close enough to a plausible honest
        update to avoid raising suspicion while gaming the temporal axis
        of the defense. The delay/staleness side of this attack sits in
        the general async-Byzantine threat model of Damaskinos et al.
        (ICML 2018) and Cox et al. (2024). No logic bug found; the noise
        multiplier is now a config knob defaulting to the original value.

        .. math::
            g' = 5.0 \cdot \text{median}(g)
            \quad
            \Delta W' = \Delta W + \eta \cdot \mathbf{n}

        Falls back to :math:`5 \cdot g_{\text{honest}}` if median is
        unavailable (early rounds). :math:`\eta` defaults to 0.05
        (``config["t2_noise_norm_mult"]``).
        """
        median_g = context.get("median_g")
        if median_g is not None:
            modified_g = median_g * 5.0
        else:
            modified_g = context["honest_g_i"] * 5.0

        noise_mult = self.config.get("t2_noise_norm_mult", 0.05)
        stale_noise = torch.randn_like(honest_delta_W) * noise_mult
        modified_dW = honest_delta_W + stale_noise

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # S1: Direct poisoner
    # ------------------------------------------------------------------

    def _s1_direct_poison(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Direct poisoner: sign-flipped, scaled gradient with honest timing.

        Standard Sign-Flipping (SF) attack -- Allen-Zhu, Ebrahimianghazani,
        Li & Alistarh, ICLR 2020: :math:`g_i' = -c \cdot g_i`. The default
        c=10 already matches the exact convention Xie, Koyejo & Gupta use
        for SF under worker delays in Zeno++ (ICML 2020: "g will be
        replaced by -10g"), which is the more directly relevant citation
        given this is an async setting. No logic bug found; c is now a
        config knob defaulting to the original hardcoded value.

        .. math::
            \Delta W' = -c \cdot \Delta W, \quad c = 10
            \qquad
            g' = g_{\text{honest}}

        Note: L2 clipping will reduce the inflated norm; the sign-flip
        is the real attack vector.
        """
        scale = self.config.get("s1_sign_flip_scale", 10.0)
        modified_dW = -scale * honest_delta_W
        modified_g = context["honest_g_i"]
        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # Shared: directed-deviation search against the cosine spatial check
    # ------------------------------------------------------------------

    def _directed_epsilon_max(
        self,
        dW_flat: torch.Tensor,
        ref: torch.Tensor,
        ref_norm: torch.Tensor,
        r_hat: torch.Tensor,
        theta_target: float,
    ) -> float:
        r"""Largest :math:`\varepsilon \geq 0` such that

        .. math::
            \cos(\Delta W - \varepsilon \hat{\mathbf{r}},\; \mathbf{r})
            \;\geq\; \theta_{\text{target}}

        i.e. the maximum push directly against the trust anchor
        :math:`\mathbf{r}` that still clears the target cosine floor.
        Used by both S2_MIMICRY and ADAPTIVE.

        Why this is sound: writing :math:`a = \langle \Delta W,
        \hat{\mathbf{r}}\rangle`, :math:`D = \|\Delta W\|`, and
        :math:`f(\varepsilon) = \cos(\Delta W - \varepsilon\hat{\mathbf{r}},
        \mathbf{r})`, one can show

        .. math::
            f'(\varepsilon) \;\propto\; a^2 - D^2 \;\leq\; 0
            \quad \text{(Cauchy-Schwarz)}

        for every :math:`\varepsilon \geq 0` -- the sign of the derivative
        does not depend on :math:`\varepsilon` at all, so :math:`f` is
        monotonically non-increasing on the whole domain and a bisection
        is guaranteed to find the true boundary. This is *not* true for a
        perturbation direction with any component orthogonal to
        :math:`\hat{\mathbf{r}}` (see the module changelog) -- monotonicity
        specifically requires moving exactly along :math:`\pm\hat{\mathbf{r}}`.

        Returns 0.0 if even :math:`\varepsilon = 0` (the unmodified honest
        update) fails to clear ``theta_target``.
        """
        def cos_at(eps: float) -> float:
            candidate = dW_flat - eps * r_hat
            return (
                torch.dot(candidate, ref) / (torch.norm(candidate) * ref_norm + 1e-8)
            ).item()

        if cos_at(0.0) < theta_target:
            return 0.0

        dW_norm = torch.norm(dW_flat).item()
        lo, hi = 0.0, max(dW_norm, 1.0) * 2.0
        # Exponentially expand the bracket so the bisection below is valid
        # regardless of how permissive theta_target is (a fixed 2x bound
        # is not guaranteed to bracket the root for every theta_cos).
        while cos_at(hi) >= theta_target and hi < 1e6:
            hi *= 2.0

        for _ in range(MAX_BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2
            if cos_at(mid) >= theta_target:
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # S2: Directed-deviation ("mimicry") poisoner
    # ------------------------------------------------------------------

    def _s2_mimicry(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Directed-deviation poisoner: the maximum push directly against
        the Top-K trust anchor that still clears
        :math:`\theta_{\cos} + \text{margin}`.

        Cosine-constrained instance of the Min-Max/Min-Sum optimization
        framework (Shejwalkar & Houmansadr, NDSS 2021): craft the update
        as :math:`\Delta W + \varepsilon \mathbf{v}` along a fixed
        adversarial direction, with :math:`\varepsilon` found by
        searching for the boundary of the defense's acceptance region.
        We use their empirically-strongest "inverse unit vector"
        direction :math:`\mathbf{v} = -\hat{\mathbf{r}}` against this
        defense's cosine constraint (see ``_directed_epsilon_max`` for
        why this direction -- and only this direction -- keeps the
        search provably sound). See the class changelog for what this
        replaces and why the old version was broken about half the time.

        .. math::
            \Delta W' = \Delta W - \varepsilon^{*} \hat{\mathbf{r}},
            \qquad
            \varepsilon^{*} = \max\{\varepsilon \geq 0 :
            \cos(\Delta W - \varepsilon\hat{\mathbf{r}}, \mathbf{r})
            \geq \theta_{\cos} + \delta\}
        """
        modified_g = context["honest_g_i"]

        ref_dW: Optional[torch.Tensor] = context.get("ref_delta_W")
        if ref_dW is None:
            return (honest_delta_W, modified_g)

        ref = ref_dW.flatten()
        ref_norm = torch.norm(ref)
        if ref_norm < 1e-8:
            return (honest_delta_W, modified_g)

        r_hat = ref / ref_norm
        dW_flat = honest_delta_W.flatten()

        margin = self.config.get("s2_cosine_margin", 0.05)
        theta_target = context["theta_cos"] + margin

        epsilon = self._directed_epsilon_max(dW_flat, ref, ref_norm, r_hat, theta_target)
        modified_dW = (dW_flat - epsilon * r_hat).reshape(honest_delta_W.shape)

        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # Adaptive adversary
    # ------------------------------------------------------------------

    def _adaptive_adversary(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        r"""Adaptive adversary: modulates timing based on P_i trend, and
        uses the same directed-deviation spatial payload as S2 with a
        small, *verified* orthogonal jitter layered on top each round so
        the attack direction is not perfectly static (evading a
        trajectory-consistency style defense signal) without reopening
        the monotonicity bug S2 had. See the class changelog.

        Timing rule (unchanged, no bug found):
        - :math:`\Delta P < -0.05` (falling): slow to :math:`1.2 \cdot g` (recover pace)
        - :math:`\Delta P > +0.05` (recovering): speed up to :math:`0.8 \cdot g`
        - otherwise: maintain :math:`g_{\text{honest}}`

        Spatial rule: find :math:`\varepsilon_{\max}` exactly as in S2,
        back off to ``adaptive_epsilon_safety`` (default 0.9) of it to
        leave headroom, then add a jitter vector orthogonal to the trust
        anchor, scaled to a fraction of the epsilon used
        (``adaptive_morph_jitter``, default 0.3). The jittered candidate's
        cosine similarity is checked before being committed; if the
        jitter would drop it below the target, the un-jittered candidate
        is used instead for that round.

        Evaluating a defense against an attacker with knowledge of the
        defense mechanism (this function) is the methodology argued for
        in Shejwalkar, Houmansadr, Kairouz & Ramage, "Back to the Drawing
        Board", IEEE S&P 2022.
        """
        self._own_P_history.append(context["own_P_i"])

        if len(self._own_P_history) < 2:
            modified_g = context["honest_g_i"]
        else:
            delta_P = self._own_P_history[-1] - self._own_P_history[-2]
            if delta_P < -0.05:
                # P_i falling -- back off and act honest temporally to recover
                modified_g = context["honest_g_i"] * 1.2
            elif delta_P > 0.05:
                # P_i recovering -- push the boundaries (speed up)
                modified_g = context["honest_g_i"] * 0.8
            else:
                modified_g = context["honest_g_i"]

        ref_dW: Optional[torch.Tensor] = context.get("ref_delta_W")
        if ref_dW is None:
            return (honest_delta_W, modified_g)

        ref = ref_dW.flatten()
        ref_norm = torch.norm(ref)
        if ref_norm < 1e-8:
            return (honest_delta_W, modified_g)

        r_hat = ref / ref_norm
        dW_flat = honest_delta_W.flatten()

        margin = self.config.get("s2_cosine_margin", 0.05)
        theta_target = context["theta_cos"] + margin

        epsilon_max = self._directed_epsilon_max(dW_flat, ref, ref_norm, r_hat, theta_target)
        if epsilon_max <= 0.0:
            return (honest_delta_W, modified_g)

        safety = self.config.get("adaptive_epsilon_safety", 0.9)
        eps_use = epsilon_max * safety
        candidate = dW_flat - eps_use * r_hat

        jitter_frac = self.config.get("adaptive_morph_jitter", 0.3)
        jitter = torch.randn_like(dW_flat)
        jitter = jitter - torch.dot(jitter, r_hat) * r_hat  # project orthogonal to r_hat
        jitter_norm = torch.norm(jitter)
        if jitter_norm > 1e-8:
            jittered = candidate + jitter / jitter_norm * (eps_use * jitter_frac)
            sim = torch.dot(jittered, ref) / (torch.norm(jittered) * ref_norm + 1e-8)
            if sim.item() >= theta_target:
                candidate = jittered
            # else: jitter would break the margin this round, keep the
            # un-jittered (verified-safe) candidate instead.

        modified_dW = candidate.reshape(honest_delta_W.shape)
        return (modified_dW, modified_g)

    # ------------------------------------------------------------------
    # Compound attack
    # ------------------------------------------------------------------

    def _compound(
        self, honest_delta_W: torch.Tensor, context: dict,
    ) -> Tuple[torch.Tensor, float]:
        """Compound: alternates between T1 spam and S2 directed-deviation.

        Note: ``self._round`` is incremented before dispatch, so round 1
        is odd (S2) and round 2 is even (T1). Both branches inherit their
        respective fixes automatically since this just calls them.
        """
        if self._round % 2 == 0:
            return self._t1_high_freq(honest_delta_W, context)
        else:
            return self._s2_mimicry(honest_delta_W, context)