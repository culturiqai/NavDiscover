# An AI-Guided Framework for Discovering and Amplifying Finite-Time Singularities in 3D Navier-Stokes Flows

This repository contains the source code and manuscript for a computational investigation into the formation of finite-time singularities in the 3D incompressible Navier-Stokes equations.

## Abstract

The question of whether the three-dimensional, incompressible Navier-Stokes equations can develop finite-time singularities from smooth initial conditions remains one of the most profound open problems in mathematics. While a formal proof is elusive, computational investigations provide critical evidence and intuition. In this work, we present a novel computational framework that goes beyond direct simulation. We employ an AI-driven discovery engine, based on an evolutionary algorithm, to systematically find optimal spectral filters that, when applied to initial conditions, maximally amplify the growth of singularity proxies in low-resolution test simulations. This framework was applied to the classic case of anti-parallel vortices, using the variable vortex separation distance as a key parameter. Our results reveal a complex, non-linear relationship between the initial geometry and the estimated blow-up time, identifying a distinct "sweet spot" for singularity formation. This work provides a powerful, data-driven methodology for mapping the landscape of potential Navier-Stokes singularities and offers a concrete, highly-unstable set of initial conditions as a target for future analytical study.

## Contents

- `paper.html`: The full manuscript of the research paper.
- `requirements.txt`: A list of the required Python libraries.
- `clay.py`: The initial 2D proof-of-concept solver.
- `clay_3d.py`: The core 3D solver with the AI-guided discovery module.
- `hypo-test.py`: The script for the initial, coarse-grained parameter sweep.
- `hypo-xtnd.py`: The script for the final, fine-grained parameter sweep that produced the main results.
- `*.png`: Image files containing the figures and results from the various experiments.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Final Experiment:**
    To reproduce the main results presented in the paper, run the extended hypothesis testing script:
    ```bash
    python3 hypo-xtnd.py
    ```
    This will run the full pipeline, including the AI-driven filter discovery and the high-resolution simulations for the parameter sweep. It will generate individual plots for each run and the final summary figure (`hypo_xtnd_sweep_summary.png`). 