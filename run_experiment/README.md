# Run experiments (reproducibility)

Run Part 1 or Part 2–3 from the **repository root**. All outputs go under **`outputs/`**:

```bash
# Part 1: CIFAR-10 cGAN ablation (PyTorch, seed=17) → outputs/part1/
python run_experiment/run_part1.py

# Part 2–3: multi-dataset pipeline (TensorFlow, seed=42) → outputs/part23/
python run_experiment/run_part23.py --dataset_key mnist
python run_experiment/run_part23.py --dataset_key sports_ball --label_noise_ps "0.0,0.1,0.2,0.3"
```

See main [README.md](../README.md) for dependencies and full options.
