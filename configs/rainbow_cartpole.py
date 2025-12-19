config = {
    # Environment / architecture
    "env_name": "ALE/Frostbite-v5",
    "network": "cnn",   # "cnn" for Atari, "mlp" for classic control

    # Optimization
    "batch_size": 32,
    "gamma": 0.99,
    "lr": 1e-4,
    "tau": 0.005,

    # C51 distribution
    "v_min": -100.0,
    "v_max": 300.0,
    "atom_size": 31,

    # Replay (PER + n-step)
    "memory_size": 100_000,
    "n_step": 3,

    # Target network updates
    "target_update_type": "hard",   # "hard" or "soft"
    "target_update_freq": 5_000     # only used for hard updates
}
