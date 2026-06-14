import numpy as np


class History:
    def __init__(self):
        self.metrics_distributed = {"accuracy": []}
        self.losses_distributed = []


def fedavg_aggregate(results, round_num=None, prev_params=None):
    results = [(p, n, m) for p, n, m in results if n > 0]
    total_examples = sum(n for _, n, _ in results)
    aggregated = None
    for params, num_examples, _ in results:
        w = num_examples / total_examples
        if aggregated is None:
            aggregated = [w * p for p in params]
        else:
            aggregated = [agg + w * p for agg, p in zip(aggregated, params)]
    return aggregated


def simulate(client_fn, num_clients, num_rounds, aggregate_fn=None, eval_fn=None):
    if aggregate_fn is None:
        aggregate_fn = fedavg_aggregate

    clients = [client_fn(str(i)) for i in range(num_clients)]
    global_params = clients[0].get_parameters(config={})
    history = History()

    for round_num in range(1, num_rounds + 1):
        prev_params = global_params  # saved before training for CS-Agg update vectors

        fit_results = []
        for client in clients:
            params, num_examples, metrics = client.fit(
                global_params, config={"round": round_num}
            )
            fit_results.append((params, num_examples, metrics))

        global_params = aggregate_fn(
            fit_results, round_num=round_num, prev_params=prev_params
        )

        if eval_fn is not None:
            acc, auc = eval_fn(global_params)
        else:
            eval_results = []
            for client in clients:
                loss, num_examples, metrics = client.evaluate(global_params, config={})
                eval_results.append((loss, num_examples, metrics))
            eval_results = [(l, n, m) for l, n, m in eval_results if n > 0]
            total = sum(n for _, n, _ in eval_results)
            acc = sum(m["accuracy"] * n for _, n, m in eval_results) / total
            def _safe_auc(m):
                v = m.get("auc", 0.5)
                return v if (v == v) else 0.5
            auc = sum(_safe_auc(m) * n for _, n, m in eval_results) / total

        history.metrics_distributed["accuracy"].append((round_num, acc))
        history.metrics_distributed.setdefault("auc", []).append((round_num, auc))
        print(f"Round {round_num:>3}/{num_rounds} | accuracy: {acc:.4f} | auc: {auc:.4f}")

    return history
