import numpy as np
from simulate import simulate, fedavg_aggregate


def make_csagg_aggregate(warmup_rounds=10):
    def aggregate(results, round_num=None, prev_params=None):
        results = [(p, n, m) for p, n, m in results if n > 0]

        if round_num is None or round_num <= warmup_rounds or prev_params is None:
            return fedavg_aggregate(results)

        # Compute update vectors: local_params - global_params_before_round
        updates = []
        for params, _, _ in results:
            update = np.concatenate([
                (p - g).flatten()
                for p, g in zip(params, prev_params)
            ])
            updates.append(update)

        # Trust score for client i = mean cosine similarity with all other clients.
        # Noisy clients produce updates misaligned with the majority -> lower score.
        n = len(updates)
        trust_scores = np.zeros(n)
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                norm_i = np.linalg.norm(updates[i])
                norm_j = np.linalg.norm(updates[j])
                if norm_i > 1e-10 and norm_j > 1e-10:
                    cos_sim = np.dot(updates[i], updates[j]) / (norm_i * norm_j)
                else:
                    cos_sim = 0.0
                sims.append(cos_sim)
            trust_scores[i] = np.mean(sims) if sims else 0.0

        # Shift to non-negative and normalize
        trust_scores = trust_scores - trust_scores.min() + 1e-8
        trust_scores = trust_scores / trust_scores.sum()

        aggregated = None
        for i, (params, _, _) in enumerate(results):
            if aggregated is None:
                aggregated = [trust_scores[i] * p for p in params]
            else:
                aggregated = [agg + trust_scores[i] * p
                               for agg, p in zip(aggregated, params)]
        return aggregated

    return aggregate


def run_csagg(client_fn, num_clients=20, num_rounds=100, warmup_rounds=10, eval_fn=None):
    aggregate_fn = make_csagg_aggregate(warmup_rounds=warmup_rounds)
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=aggregate_fn, eval_fn=eval_fn)
