# ApproxGuard

Implementation of **ApproxGuard: Aging-Aware Layerwise Mapping of Approximate Multipliers for Reliable TinyML VLSI Accelerators**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data

The bundled two-layer, two-multiplier profile declares PVT/aging grids, energy/area coefficients and an additive accuracy model. Replace these with application profiles using the same JSON schema.

## Run

```bash
approxguard execute --input examples/finite_profile.json --output runs/full --variant full
```

## Ablations

Use the same input profile for each variant: `no_aging`, `no_sensitivity`.

```bash
approxguard execute --input examples/finite_profile.json --output runs/no_aging --variant no_aging
```

## Outputs

`runs/full/mapping.json` stores selected multiplier mapping, modeled accuracy/timing and feasibility. Input and output paths are controlled by `--input` and `--output`.
