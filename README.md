# Reinforcement Learning for FTL Garbage Collection

This project implements a flash translation layer (FTL) garbage-collection simulator and compares baseline greedy and random victim-selection policies against a reinforcement-learning-based policy. It is designed to benchmark write amplification factor (WAF) and garbage-collection behavior under different workloads, with support for a Python-based PPO agent over an IPC bridge.

## How to run

### Prerequisites
- A C compiler such as GCC or clang
- Python 3 with the required packages if you want to run the RL agent

### Build the benchmark
From the project root, run:

```bash
make
```

### Run the baseline benchmark
```bash
./ftl_bench
```

### Run the RL-enabled benchmark
```bash
./ftl_bench --rl
```

### Run the Python RL agent
In a separate terminal:

```bash
python scripts/train_ppo.py
```

### Optional training loop
```bash
python scripts/train_loop.py --episodes 3 --n-writes 6000
```

## Results

These results were collected from the current implementation with the Python PPO agent connected via socket-based IPC.

- Greedy GC: WAF = 1.1842, GC_runs = 496
- Random GC: WAF = 1.5318, GC_runs = 709
- RL/PPO: WAF = 15.1029, GC_runs = 9014

The RL policy is currently under-performing compared to the baselines. Further training iterations and hyperparameter tuning are recommended to improve the agent's performance.

<<<<<<< HEAD
The RL policy is currently under-performing compared to the baselines. Further training iterations and hyperparameter tuning are recommended to improve the agent's performance.
=======
>>>>>>> 9c8b72134db5a296314a489711c3d5c819fdcb1d
