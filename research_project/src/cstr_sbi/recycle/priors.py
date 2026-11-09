"""Prior distributions for the Wu 2003 recycle CSTR 5-D inference problem.

Parameter vector:
    theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]

All multiplicative factors (alpha, beta_r, eta_col, xi_reb) equal 1.0 at
healthy nominal operation; z_A0_eff equals Z0_NOM = 0.90.
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.recycle.physics import PARAM_NAMES, Z0_NOM

# Optional torch import, guarded so this module stays importable (bounds,
# theta_to_u/u_to_theta, etc.) without torch installed. `TruncatedLogNormalPrior`/
# `InducedLogitPrior` below are defined at MODULE level (not nested inside their
# factory functions) specifically so instances are picklable -- a nested/local
# class is not (found 2026-09-04: `pickle.dump` on an SNPE posterior built with
# `truncated_lognormal_5d()` as its declared prior failed with
# "Can't pickle local object 'truncated_lognormal_5d.<locals>.TruncatedLogNormalPrior'"
# when the classes were nested; every trained-posterior .pkl in this project
# needs its prior to be picklable).
try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Prior bounds
# ---------------------------------------------------------------------------

# alpha: catalyst/kinetic activity (decay lowers rate)
ALPHA_LOW:   float = 0.40
ALPHA_HIGH:  float = 1.20

# beta_r: CSTR jacket heat-transfer fouling (lower = worse)
BETA_R_LOW:  float = 0.40
BETA_R_HIGH: float = 1.20

# eta_col: distillation column tray efficiency (lower = worse separation)
ETA_COL_LOW:  float = 0.50
ETA_COL_HIGH: float = 1.00

# xi_reb: reboiler duty scale factor (1.0 = nominal)
XI_REB_LOW:  float = 0.40
XI_REB_HIGH: float = 1.20

# z_A0_eff: effective fresh-feed A composition (nominal = 0.90)
Z_A0_LOW:  float = float(Z0_NOM) - 0.20   # 0.70
Z_A0_HIGH: float = float(Z0_NOM) + 0.05   # 0.95

PRIOR_LOW_5D = np.array(
    [ALPHA_LOW, BETA_R_LOW, ETA_COL_LOW, XI_REB_LOW, Z_A0_LOW],
    dtype=np.float32,
)
PRIOR_HIGH_5D = np.array(
    [ALPHA_HIGH, BETA_R_HIGH, ETA_COL_HIGH, XI_REB_HIGH, Z_A0_HIGH],
    dtype=np.float32,
)

# Fault classification thresholds
HEALTHY_THRESHOLDS = {
    "alpha":      0.85,
    "beta_r":     0.85,
    "eta_col":    0.80,
    "xi_reb_lo":  0.85,   # below this = reboiler starvation
    "xi_reb_hi":  1.15,   # above this = reboiler flooding
    "z_A0_dev":   0.05,   # |z_A0 - 0.90| > this = feed composition fault
}


# ---------------------------------------------------------------------------
# Truncated-lognormal prior, centered at nominal healthy operation
# ---------------------------------------------------------------------------
# This has been the project's *actual* training prior since notebooks/37 (all of
# nb37/38/39/24a use it), but until now it existed only as copy-pasted inline
# classes in notebooks and in `/tmp` scripts (wiped on reboot). Library-ised here
# per `system_ii_sbi_solidification_plan.md` Stage 0.2 -- import from here going
# forward instead of re-pasting the class.

NOMINAL_THETA_5D = np.array([1.0, 1.0, 1.0, 1.0, 0.90], dtype=np.float32)
LOGN_SIGMA_5D = np.array([0.16, 0.16, 0.10, 0.16, 0.055], dtype=np.float32)

EPS_LOGIT_BOX: float = 1e-5


def theta_to_u(theta, low=PRIOR_LOW_5D, high=PRIOR_HIGH_5D, eps=EPS_LOGIT_BOX):
    """Map theta (box support) to unconstrained logit-space u. Inverse of `u_to_theta`."""
    z = np.clip((np.asarray(theta, dtype=np.float32) - low) / (high - low), eps, 1.0 - eps)
    return np.log(z / (1.0 - z)).astype(np.float32)


def u_to_theta(u, low=PRIOR_LOW_5D, high=PRIOR_HIGH_5D):
    """Map unconstrained logit-space u back to theta (box support). Inverse of `theta_to_u`."""
    z = 1.0 / (1.0 + np.exp(-np.asarray(u, dtype=np.float32)))
    return (low + z * (high - low)).astype(np.float32)


if torch is not None:

    class TruncatedLogNormalPrior(torch.distributions.Distribution):
        """Truncated-lognormal prior over the 5-D theta box, centered at `median`.

        Same distribution class used inline in notebooks/37-39/24a (kept
        numerically identical -- do not change the sampling/log_prob logic
        without re-validating against those notebooks' cached posteriors).
        Defined at MODULE level (not nested in `truncated_lognormal_5d`) so
        instances -- and anything holding a reference to one, e.g. a trained
        SNPE posterior -- are picklable. See the module-level comment above
        the `torch` import for why this matters.
        """
        arg_constraints = {}
        has_rsample = False

        def __init__(self, median, sigma, low, high, validate_args=False):
            self.median = torch.as_tensor(median, dtype=torch.float32)
            self.sigma = torch.as_tensor(sigma, dtype=torch.float32)
            self.low = torch.as_tensor(low, dtype=torch.float32)
            self.high = torch.as_tensor(high, dtype=torch.float32)
            self.mu = torch.log(self.median)
            self.base = torch.distributions.Normal(self.mu, self.sigma)
            lo_z = (torch.log(self.low) - self.mu) / self.sigma
            hi_z = (torch.log(self.high) - self.mu) / self.sigma
            standard = torch.distributions.Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu))
            self.log_norm = torch.log(standard.cdf(hi_z) - standard.cdf(lo_z)).sum()
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([len(self.median)]),
                              validate_args=validate_args)

        @property
        def mean(self):
            draws = self.sample((20000,))
            return draws.mean(dim=0)

        @property
        def variance(self):
            draws = self.sample((20000,))
            return draws.var(dim=0)

        def sample(self, sample_shape=torch.Size()):
            if isinstance(sample_shape, int):
                sample_shape = torch.Size([sample_shape])
            sample_shape = torch.Size(sample_shape)
            n = int(np.prod(sample_shape)) if len(sample_shape) else 1
            accepted = []
            batch = max(1024, 2 * n)
            while sum(a.shape[0] for a in accepted) < n:
                theta = torch.exp(self.base.sample((batch,)))
                mask = ((theta >= self.low) & (theta <= self.high)).all(dim=1)
                if mask.any():
                    accepted.append(theta[mask])
            out = torch.cat(accepted, dim=0)[:n]
            return out.reshape(sample_shape + self.event_shape)

        def log_prob(self, value):
            value = torch.as_tensor(value, dtype=torch.float32)
            inside = ((value >= self.low) & (value <= self.high)).all(dim=-1)
            safe_value = torch.clamp(value, min=torch.finfo(torch.float32).tiny)
            logpdf = (self.base.log_prob(torch.log(safe_value)).sum(dim=-1)
                      - torch.log(safe_value).sum(dim=-1) - self.log_norm)
            return torch.where(inside, logpdf, torch.full_like(logpdf, -torch.inf))


    class InducedLogitPrior(torch.distributions.Distribution):
        """Exact induced density of `theta_prior` pushed through `theta_to_u`.

        Analytic change-of-variables pushforward (not an empirical-Gaussian
        surrogate) -- so SNPE's declared `prior` in logit space matches the
        density the training bank was actually drawn from. Same class as
        notebooks/39 Part II. Defined at MODULE level for the same
        picklability reason as `TruncatedLogNormalPrior` above.
        """
        arg_constraints = {}
        has_rsample = False

        def __init__(self, theta_prior, low, high, eps=EPS_LOGIT_BOX, validate_args=False):
            self.theta_prior = theta_prior
            self.low = torch.as_tensor(low, dtype=torch.float32)
            self.high = torch.as_tensor(high, dtype=torch.float32)
            self.span = self.high - self.low
            self.eps = eps
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([len(self.low)]),
                              validate_args=validate_args)

        def _u_to_theta_and_z(self, u):
            z = torch.sigmoid(torch.as_tensor(u, dtype=torch.float32))
            return self.low + z * self.span, z

        def sample(self, sample_shape=torch.Size()):
            theta = self.theta_prior.sample(sample_shape)
            z = torch.clamp((theta - self.low) / self.span, self.eps, 1.0 - self.eps)
            return torch.log(z / (1.0 - z))

        def log_prob(self, value):
            theta, z = self._u_to_theta_and_z(value)
            log_jacobian = torch.log(self.span).sum() + torch.log(z * (1.0 - z) + 1e-12).sum(dim=-1)
            return self.theta_prior.log_prob(theta) + log_jacobian


def truncated_lognormal_5d(median=NOMINAL_THETA_5D, sigma=LOGN_SIGMA_5D,
                            low=PRIOR_LOW_5D, high=PRIOR_HIGH_5D):
    """Construct a `TruncatedLogNormalPrior` instance. Requires torch."""
    if torch is None:
        raise ImportError("torch must be installed. Run: pip install torch")
    return TruncatedLogNormalPrior(median, sigma, low, high)


def induced_logit_prior(theta_prior, low=PRIOR_LOW_5D, high=PRIOR_HIGH_5D, eps=EPS_LOGIT_BOX):
    """Construct an `InducedLogitPrior` instance. Requires torch."""
    if torch is None:
        raise ImportError("torch must be installed. Run: pip install torch")
    return InducedLogitPrior(theta_prior, low, high, eps)


if torch is not None:

    class MixedAlphaUniformPrior(torch.distributions.Distribution):
        """5-D prior: `alpha` ~ Uniform(low, high); `(beta_r, eta_col, xi_reb,
        z_A0_eff)` ~ their existing truncated-lognormal marginals, independent
        of `alpha` given the box truncation's per-dimension separability (the
        same property `TruncatedLogNormalPrior`'s marginals already rely on).

        Motivated 2026-09-05: `alpha` was found to be the ONLY parameter with a
        catastrophic, seed-dependent shrinkage-toward-prior-mean failure across
        every S-C.2 variant tested this session (monolithic, factorised,
        architecture probe) -- traced to the Wu2003 named fault scenarios
        sitting 1-3.4 log-sigma into the narrow lognormal prior's tail for
        `alpha` specifically, a region the training bank barely covers.
        Widening ONLY `alpha` (not all 5 dimensions, as the rejected
        full-uniform-prior attempt did) targets the diagnosed problem directly,
        at much lower expected cost: the full-uniform bank's accept rate
        collapsed to ~37% (vs ~91-95% for the all-lognormal bank) because
        MULTIPLE parameters going simultaneously extreme at once is what makes
        the matched-protocol warm-start solve hard to converge -- the real
        Wu2003 scenarios are mostly single/double-fault, so widening only
        `alpha` should keep the other four parameters' well-behaved narrow
        range intact and avoid that blowup.
        """
        arg_constraints = {}
        has_rsample = False

        def __init__(self, alpha_low, alpha_high, median_rest, sigma_rest, low_rest, high_rest,
                     validate_args=False):
            self.alpha_dist = torch.distributions.Uniform(
                torch.as_tensor(alpha_low, dtype=torch.float32),
                torch.as_tensor(alpha_high, dtype=torch.float32),
                validate_args=False,  # else log_prob raises instead of returning -inf out-of-support
            )
            self.rest_dist = TruncatedLogNormalPrior(median_rest, sigma_rest, low_rest, high_rest)
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([5]),
                              validate_args=validate_args)

        def sample(self, sample_shape=torch.Size()):
            if isinstance(sample_shape, int):
                sample_shape = torch.Size([sample_shape])
            sample_shape = torch.Size(sample_shape)
            alpha_samp = self.alpha_dist.sample(sample_shape).unsqueeze(-1)
            rest_samp = self.rest_dist.sample(sample_shape)
            return torch.cat([alpha_samp, rest_samp], dim=-1)

        def log_prob(self, value):
            value = torch.as_tensor(value, dtype=torch.float32)
            alpha_logp = self.alpha_dist.log_prob(value[..., 0])
            rest_logp = self.rest_dist.log_prob(value[..., 1:])
            return alpha_logp + rest_logp


def mixed_alpha_uniform_prior(alpha_low=ALPHA_LOW, alpha_high=ALPHA_HIGH,
                                median_rest=None, sigma_rest=None,
                                low_rest=None, high_rest=None):
    """Construct a `MixedAlphaUniformPrior` instance (alpha uniform, the other
    4 parameters their existing lognormal marginals). Requires torch.

    `median_rest`/`sigma_rest`/`low_rest`/`high_rest` default to
    `NOMINAL_THETA_5D`/`LOGN_SIGMA_5D`/`PRIOR_LOW_5D`/`PRIOR_HIGH_5D` sliced to
    indices `[1:]` (`beta_r, eta_col, xi_reb, z_A0_eff` -- `alpha` is index 0
    in `PARAM_NAMES`).
    """
    if torch is None:
        raise ImportError("torch must be installed. Run: pip install torch")
    if median_rest is None:
        median_rest = NOMINAL_THETA_5D[1:]
    if sigma_rest is None:
        sigma_rest = LOGN_SIGMA_5D[1:]
    if low_rest is None:
        low_rest = PRIOR_LOW_5D[1:]
    if high_rest is None:
        high_rest = PRIOR_HIGH_5D[1:]
    return MixedAlphaUniformPrior(alpha_low, alpha_high, median_rest, sigma_rest, low_rest, high_rest)


if torch is not None:

    class MixedWidePrior(torch.distributions.Distribution):
        """5-D prior generalizing `MixedAlphaUniformPrior`: an arbitrary
        subset of dimensions ~ independent Uniform over their prior box, the
        remaining dimensions ~ their existing truncated-lognormal marginals
        (jointly independent, same box-truncation separability argument as
        `MixedAlphaUniformPrior`/`TruncatedLogNormalPrior`).

        Motivated 2026-09-05 (`system_ii_sbi_solidification_plan.md`
        priority-0 follow-up): `alpha`'s narrow-lognormal-prior/Wu2003-fault-
        catalogue coverage mismatch was fixed by widening it alone
        (`MixedAlphaUniformPrior`, see that class's docstring). The same
        coverage argument applies to `beta_r`, `eta_col`, `xi_reb` -- their
        own named-fault values (e.g. `beta_r` down to 0.60, `xi_reb` down to
        0.70) sit multiple log-sigma into the narrow lognormal core too --
        but NOT to `z_A0_eff`: its near-exact dedicated feed-analyzer signal
        makes its posterior largely prior-width-insensitive, so keep it
        narrow to limit the joint accept-rate cost. This class widens an
        arbitrary index subset (default: everything except `z_A0_eff`,
        index 4) rather than hardcoding the single-`alpha` case.
        **Caution**: a full 5-D uniform prior was already tried and aborted
        (accept rate collapsed to ~37%, multiple simultaneously-extreme
        parameters make the matched-protocol warm-start solve hard to
        converge). Widening 4/5 dims is a new, not-yet-quantified
        intermediate point between that failure and the single-`alpha`
        success (~75% accept rate) -- smoke-test the accept rate on a
        handful of draws before committing to a full bank generation.

        Generalized 2026-09-10 to arbitrary dimensionality (`n_dims = len(low)`,
        not hardcoded 5) so the same class also serves the strict 4-D `S-B3`
        scope (no `z_A0_eff`), see `cstr_sbi.recycle.sb3.mixed_wide_prior_4d_sb3`.
        """
        arg_constraints = {}
        has_rsample = False

        def __init__(self, uniform_idx, low, high, median, sigma, validate_args=False):
            uniform_idx = sorted(set(int(i) for i in uniform_idx))
            n_dims = len(low)
            lognormal_idx = [i for i in range(n_dims) if i not in uniform_idx]
            self.n_dims = n_dims
            self.uniform_idx = uniform_idx
            self.lognormal_idx = lognormal_idx
            low_t = torch.as_tensor(low, dtype=torch.float32)
            high_t = torch.as_tensor(high, dtype=torch.float32)
            self.uniform_dist = torch.distributions.Uniform(
                low_t[uniform_idx], high_t[uniform_idx], validate_args=False,
            )
            if lognormal_idx:
                median_t = torch.as_tensor(median, dtype=torch.float32)
                sigma_t = torch.as_tensor(sigma, dtype=torch.float32)
                self.lognormal_dist = TruncatedLogNormalPrior(
                    median_t[lognormal_idx], sigma_t[lognormal_idx],
                    low_t[lognormal_idx], high_t[lognormal_idx],
                )
            else:
                self.lognormal_dist = None
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([n_dims]),
                              validate_args=validate_args)

        def sample(self, sample_shape=torch.Size()):
            if isinstance(sample_shape, int):
                sample_shape = torch.Size([sample_shape])
            sample_shape = torch.Size(sample_shape)
            out = torch.empty(sample_shape + (self.n_dims,), dtype=torch.float32)
            out[..., self.uniform_idx] = self.uniform_dist.sample(sample_shape)
            if self.lognormal_dist is not None:
                out[..., self.lognormal_idx] = self.lognormal_dist.sample(sample_shape)
            return out

        def log_prob(self, value):
            value = torch.as_tensor(value, dtype=torch.float32)
            logp = self.uniform_dist.log_prob(value[..., self.uniform_idx]).sum(dim=-1)
            if self.lognormal_dist is not None:
                logp = logp + self.lognormal_dist.log_prob(value[..., self.lognormal_idx])
            return logp


def mixed_wide_prior(uniform_idx=(0, 1, 2, 3), low=PRIOR_LOW_5D, high=PRIOR_HIGH_5D,
                      median=NOMINAL_THETA_5D, sigma=LOGN_SIGMA_5D):
    """Construct a `MixedWidePrior` instance. Default widens `alpha, beta_r,
    eta_col, xi_reb` (indices 0-3 in `PARAM_NAMES`) to independent
    `Uniform(low, high)` over their prior box, and keeps `z_A0_eff` (index 4)
    as its existing narrow truncated-lognormal. Requires torch.
    """
    if torch is None:
        raise ImportError("torch must be installed. Run: pip install torch")
    return MixedWidePrior(uniform_idx, low, high, median, sigma)


def box_uniform_5d():
    """5-D BoxUniform prior for sbi / SNPE_C.

    Returns a ``sbi.utils.BoxUniform`` over
    [alpha, beta_r, eta_col, xi_reb, z_A0_eff].
    """
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:
        raise ImportError(
            "sbi and torch must be installed. Run: pip install sbi torch"
        ) from e

    return BoxUniform(
        low=torch.tensor(PRIOR_LOW_5D,  dtype=torch.float32),
        high=torch.tensor(PRIOR_HIGH_5D, dtype=torch.float32),
    )
