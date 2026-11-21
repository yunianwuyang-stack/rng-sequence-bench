"""Prefix-only categorical choice models for the revision."""

import numpy as np
from scipy.special import logsumexp

GRID = np.array([(j % 3, j // 3) for j in range(9)], dtype=float)
LINE = np.arange(9, dtype=float)[:, None]
RD = ("repeat", "distance")
RDC = (*RD, "gap")
FREQUENCY = (*RDC, "global", "local10")
MODELS = {
    "RD": RD,
    "RDC": RDC,
    "frequency": FREQUENCY,
    "memory": (*RD, "memory4"),
    "bounded": (*RD, "cap14"),
    "RDC_bias": (*RDC, "bias"),
    "frequency_bias": (*FREQUENCY, "bias"),
    "bounded_bias": (*RD, "cap14", "bias"),
}


def distances(coords):
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)


class History:
    def __init__(self, participants, coords, terms):
        self.coords = distances(coords)
        self.positions = np.asarray(coords)
        self.terms = terms
        self.people = np.arange(participants)
        self.categories = np.arange(len(coords))
        shape = (participants, len(coords))
        self.gap = np.ones(shape, dtype=int)
        self.total = np.zeros(shape, dtype=int)
        self.local = {
            int(name[5:]): np.zeros(shape, dtype=int)
            for name in terms
            if name.startswith("local")
        }
        self.memory = {
            int(name[6:]): np.zeros(shape)
            for name in terms
            if name.startswith("memory")
        }

    def update(self, sequences, t):
        last = sequences[:, t - 1]
        self.gap += 1
        self.gap[self.people, last] = 1
        self.total[self.people, last] += 1
        for window, counts in self.local.items():
            counts[self.people, last] += 1
            if t > window:
                counts[self.people, sequences[:, t - window - 1]] -= 1
        for half_life, memory in self.memory.items():
            memory *= 2 ** (-1 / half_life)
            memory[self.people, last] += 1
        features = {
            "repeat": last[:, None] == self.categories,
            "distance": self.coords[last],
            "gap": np.log2(self.gap),
            "linear_gap": self.gap,
            "inverse_gap": 1.0 / self.gap,
            "gap2to4": (self.gap >= 2) & (self.gap <= 4),
            "gap5to9": (self.gap >= 5) & (self.gap <= 9),
            "gap10to19": (self.gap >= 10) & (self.gap <= 19),
            "gap20plus": self.gap >= 20,
            "global": (t / len(self.categories) - self.total) / np.sqrt(t),
        }
        for name in self.terms:
            if name.startswith("cap"):
                features[name] = np.log2(np.minimum(self.gap, int(name[3:])))
        for window, counts in self.local.items():
            h = min(t, window)
            features[f"local{window}"] = (h / len(self.categories) - counts) / np.sqrt(
                h
            )
        features.update({f"memory{h}": value for h, value in self.memory.items()})
        if "direction" in self.terms:
            alignment = np.zeros_like(self.gap, dtype=float)
            if t > 1:
                current = self.positions[last]
                incoming = current - self.positions[sequences[:, t - 2]]
                proposed = self.positions[None] - current[:, None]
                divisor = self.coords[last] * np.linalg.norm(incoming, axis=1)[:, None]
                np.divide(
                    np.einsum("nkd,nd->nk", proposed, incoming),
                    divisor,
                    out=alignment,
                    where=divisor > 0,
                )
            features["direction"] = alignment
        return features


def design_features(sequences, coords, terms):
    requested = set(terms) - {"bias"}
    n, length = sequences.shape
    data = {name: np.empty((n, length - 1, len(coords))) for name in requested}
    history = History(n, coords, requested)
    for t in range(1, length):
        features = history.update(sequences, t)
        for name in requested:
            data[name][:, t - 1] = features[name]
    return data


def design(data, terms, trial_slice=slice(None)):
    columns = [data[name][:, trial_slice] for name in terms if name != "bias"]
    x = np.stack(columns, axis=-1)
    if "bias" in terms:
        n, t, k, _ = x.shape
        offsets = np.broadcast_to(np.eye(k), (n, t, k, k))
        x = np.concatenate((x, offsets), axis=-1)
    return x


def fit_batch(x, y, precision=1.0, center=0.0, covariance=False):
    """Independent ridge-logit Newton fits, with local inverse posterior Hessians."""
    n, t, k, d = x.shape
    precision = np.broadcast_to(np.asarray(precision), (d,))
    center = np.broadcast_to(center, (n, d))
    beta = center.copy().astype(float)
    flat = x.reshape(n, t * k, d)
    totals = x[np.arange(n)[:, None], np.arange(t), y].sum(axis=1)
    index = np.arange(d)

    def state(candidate):
        eta = np.einsum("ntkd,nd->ntk", x, candidate)
        norm = logsumexp(eta, axis=2)
        change = candidate - center
        value = norm.sum(axis=1) - (totals * candidate).sum(axis=1)
        value += ((change**2) * precision).sum(axis=1) / 2
        return value, eta, norm, change

    for _ in range(45):
        value, eta, norm, change = state(beta)
        probs = np.exp(eta - norm[:, :, None])
        means = np.einsum("ntk,ntkd->ntd", probs, x)
        grad = means.sum(axis=1) - totals + change * precision
        weighted = flat * probs.reshape(n, -1, 1)
        hessian = flat.swapaxes(1, 2) @ weighted - means.swapaxes(1, 2) @ means
        hessian[:, index, index] += precision
        if np.max(np.abs(grad)) < 5e-6:
            return (beta, np.linalg.inv(hessian)) if covariance else beta
        step = np.linalg.solve(hessian, grad[..., None])[..., 0]
        factor = np.ones(n)
        for _ in range(22):
            proposed = beta - factor[:, None] * step
            loss = state(proposed)[0]
            accepted = loss <= value + 16 * np.finfo(float).eps * np.maximum(
                1, np.abs(value)
            )
            if accepted.all():
                break
            factor[~accepted] *= 0.5
        else:
            raise RuntimeError("Newton line search did not converge")
        beta = proposed
    raise RuntimeError(f"Unconverged penalized score: {np.max(np.abs(grad)):.3g}")


def predict(x, beta):
    beta = np.broadcast_to(beta, (len(x), beta.shape[-1]))
    eta = np.einsum("ntkd,nd->ntk", x, beta)
    return np.exp(eta - logsumexp(eta, axis=-1, keepdims=True))


def precision_for(terms, ridge=1.0, bias_ridge=4.0):
    if "bias" not in terms:
        return ridge
    return np.r_[np.full(len(terms) - 1, ridge), np.full(9, bias_ridge)]


def simulate(params, terms, coords, uniforms, *, prefix=None):
    n, length = uniforms.shape
    k = len(coords)
    data = np.empty((n, length), dtype=int)
    if prefix is None:
        data[:, 0] = (uniforms[:, 0] * k).astype(int)
        start = 1
    else:
        start = prefix.shape[1]
        if prefix.shape[0] != n or not 1 <= start < length:
            raise ValueError(
                "Prefix must match participants and leave at least one generated choice"
            )
        data[:, :start] = prefix
    history = History(n, coords, terms)
    core_terms = [name for name in terms if name != "bias"]
    for t in range(1, length):
        features = history.update(data, t)
        if t < start:
            continue
        active = params if params.ndim == 2 else params[:, t - 1]
        logits = sum(
            (active[:, j, None] * features[term] for j, term in enumerate(core_terms)),
            np.zeros((n, k)),
        )
        if "bias" in terms:
            logits += active[:, len(core_terms) :]
        p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        cumulative = p.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        data[:, t] = (uniforms[:, t, None] >= cumulative).sum(axis=1)
    return data
