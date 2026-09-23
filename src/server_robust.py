import numpy as np
from simulate import simulate


def _flatten(params):
    return np.concatenate([p.flatten() for p in params])


def _unflatten_like(flat, template):
    out = []
    idx = 0
    for p in template:
        n = p.size
        out.append(flat[idx:idx + n].reshape(p.shape))
        idx += n
    return out


def make_krum_aggregate(num_byzantine):
    """Krum (Blanchard et al., NeurIPS 2017): selects the single client update
    whose sum of squared distances to its (n - f - 2) nearest neighbors is
    smallest. Requires an a priori bound f on the number of corrupted clients
    -- unlike CS-Agg, which requires no such bound."""
    def aggregate(results, round_num=None, prev_params=None):
        results = [(p, n, m) for p, n, m in results if n > 0]
        n = len(results)
        if n <= 2:
            w = 1.0 / n
            aggregated = None
            for params, _, _ in results:
                aggregated = [w * p for p in params] if aggregated is None else \
                    [agg + w * p for agg, p in zip(aggregated, params)]
            return aggregated

        f = int(min(num_byzantine, max(0, (n - 3) // 2)))
        flat = [_flatten(p) for p, _, _ in results]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum((flat[i] - flat[j]) ** 2)
                dists[i, j] = d
                dists[j, i] = d

        k = max(n - f - 2, 1)
        scores = np.zeros(n)
        for i in range(n):
            others = np.delete(dists[i], i)
            scores[i] = np.sort(others)[:k].sum()

        winner = int(np.argmin(scores))
        return results[winner][0]
    return aggregate


def make_trimmed_mean_aggregate(num_byzantine):
    """Coordinate-wise trimmed mean (Yin et al., ICML 2018): drop the f
    largest and f smallest values per coordinate, average what remains.
    Requires the same a priori bound f as Krum."""
    def aggregate(results, round_num=None, prev_params=None):
        results = [(p, n, m) for p, n, m in results if n > 0]
        n = len(results)
        f = int(min(num_byzantine, max(0, (n - 1) // 2)))
        template = results[0][0]
        stacked = np.stack([_flatten(p) for p, _, _ in results], axis=0)  # (n, d)
        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[f:n - f] if f > 0 else sorted_vals
        mean_flat = trimmed.mean(axis=0)
        return _unflatten_like(mean_flat, template)
    return aggregate


def coordinate_median_aggregate(results, round_num=None, prev_params=None):
    """Coordinate-wise median (Yin et al., ICML 2018): no a priori bound on
    the number of corrupted clients is required."""
    results = [(p, n, m) for p, n, m in results if n > 0]
    template = results[0][0]
    stacked = np.stack([_flatten(p) for p, _, _ in results], axis=0)
    median_flat = np.median(stacked, axis=0)
    return _unflatten_like(median_flat, template)


def uniform_mean_aggregate(results, round_num=None, prev_params=None):
    """Unweighted (uniform) mean of client parameters -- the control baseline
    that isolates the effect of *dropping sample-count weighting* from the
    effect of using a *robust statistic*. FedAvg weights client i by n_i;
    Krum, trimmed mean, coordinate-wise median and (post-warmup) CS-Agg all
    ignore n_i. Without this control, any advantage those methods show under
    extreme Dirichlet skew is confounded: it could come from robustness, or
    merely from not over-weighting the one client that holds most of the data."""
    results = [(p, n, m) for p, n, m in results if n > 0]
    w = 1.0 / len(results)
    aggregated = None
    for params, _, _ in results:
        aggregated = [w * p for p in params] if aggregated is None else \
            [agg + w * p for agg, p in zip(aggregated, params)]
    return aggregated


def run_uniform_mean(client_fn, num_clients=20, num_rounds=100, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=uniform_mean_aggregate, eval_fn=eval_fn)


def run_krum(client_fn, num_clients=20, num_rounds=100, num_byzantine=4, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=make_krum_aggregate(num_byzantine), eval_fn=eval_fn)


def run_trimmed_mean(client_fn, num_clients=20, num_rounds=100, num_byzantine=4, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=make_trimmed_mean_aggregate(num_byzantine), eval_fn=eval_fn)


def run_coordinate_median(client_fn, num_clients=20, num_rounds=100, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=coordinate_median_aggregate, eval_fn=eval_fn)
