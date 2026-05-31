# Flow Matching Parameterisation

**COMP4680/ENGN8650 Mini-Project 2** — Mayukh Das (U7965027), ANU

## Overview

This project investigates how prediction target (x vs v) and loss parameterisation affect flow matching models across ambient dimensions, and implements MeanFlow for one-step generation.

### Key findings

- **x-prediction** remains stable at all dimensions (D=2, 8, 32) because it only needs to model the low-dimensional data structure
- **v-prediction** fails at D=32 (target scales with ambient dimension) but can be rescued by widening the network to h=1024 (~16x compute)
- **MeanFlow** achieves clean one-step generation on swiss_roll and circles, but shows mode concentration artifacts on multi-modal data (gaussians)

## Structure

```
part1.py                        # Warm-up: data visualization + v-prediction at D=2
part2.py                        # Parameterisation: 4 pred/loss combos x 3 datasets x 3 dims
part3.py                        # Rescuing v-prediction: width x schedule sweep
part4.1.py                      # Sampling efficiency: Euler step-count sweep
part4.2.py                      # MeanFlow implementation with JVP-based training
script.py                       # Utility: composite figure for Part 2
meanflow_3x3_grid_script.py     # Utility: MeanFlow summary grid
part3_width_sweep_script.py     # Utility: Part 3 sweep composite figure
src/dataloader.py               # Dataset loader (swiss_roll, gaussians, circles)
data/                           # Pre-generated .npz datasets
part2_figures/                  # Part 2 output (36 figures)
part3_figures/                  # Part 3 output (width x schedule sweep)
part4.1_figures/                # Part 4.1 output (step-count sweep)
part4.2_figure/                 # Part 4.2 output (MeanFlow generations)
```

## Running

```bash
pip install -r requirements.txt
python part1.py
python part2.py
python part3.py
python part4.1.py
python part4.2.py
```

Each script trains from scratch and saves figures to its output directory. Parts 1-3 train for 25,000 steps; Part 4 trains for 50,000 steps. All use a 5-layer MLP with 256 hidden units (except Part 3 which sweeps width).

## Report

The submitted report is [`u7965027_MiniProject2_Flow_Matching.pdf`](u7965027_MiniProject2_Flow_Matching.pdf).

## References

- [Back to Basics (JiT)](https://arxiv.org/abs/2511.13720): prediction parameterization in flow matching
- [RAE](https://arxiv.org/abs/2510.11690): parameterization and dimension
- [MeanFlow](https://arxiv.org/abs/2505.13447): one-step generation
