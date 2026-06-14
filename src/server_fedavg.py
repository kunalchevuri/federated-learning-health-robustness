from simulate import simulate, fedavg_aggregate


def run_fedavg(client_fn, num_clients=20, num_rounds=100, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=fedavg_aggregate, eval_fn=eval_fn)
