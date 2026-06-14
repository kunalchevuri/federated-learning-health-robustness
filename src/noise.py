import numpy as np

def inject_noise(y, noise_rate, seed=42):
    """
    Asymmetric label noise:
    - depressed (1) → not depressed (0) at rate noise_rate
    - not depressed (0) → depressed (1) at rate noise_rate / 2
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()

    # flip 1 → 0
    dep_indices = np.where(y == 1)[0]
    n_flip = int(len(dep_indices) * noise_rate)
    flip_indices = rng.choice(dep_indices, size=n_flip, replace=False)
    y_noisy[flip_indices] = 0

    # flip 0 → 1
    nodep_indices = np.where(y == 0)[0]
    n_flip = int(len(nodep_indices) * (noise_rate / 2))
    flip_indices = rng.choice(nodep_indices, size=n_flip, replace=False)
    y_noisy[flip_indices] = 1

    return y_noisy


def inject_symmetric_noise(y, noise_rate, seed=42):
    """Symmetric label noise: each class flipped at the same rate."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    for label in [0, 1]:
        indices = np.where(y == label)[0]
        n_flip = int(len(indices) * noise_rate)
        if n_flip > 0:
            flip_idx = rng.choice(indices, size=n_flip, replace=False)
            y_noisy[flip_idx] = 1 - label
    return y_noisy


if __name__ == "__main__":
    # test with fake labels: 50 depressed, 50 not
    y = np.array([1] * 50 + [0] * 50)
    
    for rate in [0.1, 0.2, 0.3, 0.4]:
        y_noisy = inject_noise(y, noise_rate=rate)
        flipped = (y != y_noisy).sum()
        print(f"Noise rate {rate:.0%}: {flipped} labels flipped "
              f"({flipped/len(y):.1%} of total)")