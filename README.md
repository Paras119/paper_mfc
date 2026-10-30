# GPU-enabled 3D microbial fuel-cell model

This repository contains the CUDA/Numba implementation used for the hybrid three-dimensional agent-based microbial fuel-cell framework. The authoritative current script is `mfc_cuda_model.py`.

## Model capabilities

- Three-dimensional agent dynamics, attachment, growth, division, decay, shoving, and detachment
- Acetate and proton transport with pseudo-steady diffusion relaxation
- Single-species, competitive, and commensal configurations
- Lumped electrochemical current and voltage calculation
- Resumable one-at-a-time (OAT) sensitivity analysis
- Instantaneous polarization and power-density post-processing

## Requirements

- Python 3.10 or later
- An NVIDIA GPU with a working CUDA driver
- Packages listed in `requirements.txt`

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Polarization analysis

To run the baseline model for 240 hours and calculate the polarization curve at the resulting reactor state:

```bash
python mfc_cuda_model.py \
  --polarization \
  --sim-hours 240 \
  --seed 43 \
  --output-dir polarization_outputs
```

The command writes:

- `polarization_day10.csv`
- `polarization_day10.png`

The calculation evaluates 60 logarithmically spaced external resistances from 50,000 to 10 ohm. At each resistance, the electrical current is calculated using 40 under-relaxed fixed-point iterations. Agent states, biomass, substrate and proton fields, and biological current availability are held fixed during the resistance sweep. The resulting points are therefore instantaneous electrical post-processing calculations, not biologically or transport-equilibrated steady-state measurements.

By default, a reproducible zero-mean Gaussian voltage perturbation with a standard deviation of 0.005 V is applied after each electrical solution. Set `--polarization-noise-sd 0` to produce the unperturbed calculated curve.

## OAT sensitivity analysis

Run the default resumable sensitivity analysis with:

```bash
python mfc_cuda_model.py --all-cases --output-dir oat_outputs
```

For an inexpensive interface check:

```bash
python mfc_cuda_model.py --smoke-test --all-cases --output-dir oat_smoke_test
```

Completed cases are archived with completion markers and can be resumed without rerunning valid outputs. Use `--overwrite` only when existing completed cases should be regenerated.

## Reproducibility note

The stochastic agent model and the optional polarization voltage perturbation are controlled by the supplied seed. GPU and dependency versions should be recorded when generating archival results because floating-point behavior can vary across CUDA hardware and software versions.
