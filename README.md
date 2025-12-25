# Triskel-MC

<img src="assets/TRISKEL-MC.gif" width="450" alt="TRISKEL-MC">

Time continuous Markov Chain Monte Carlo with product space

## Installation

Install the package in editable mode to use the provided NumPy/JAX utilities:

```bash
pip install -e .
```

Import the NumPy utilities from the package namespace rather than the legacy top-level module:

```python
from triskel_mc import bd_mh_step_numpy
```

## Method overview

The main sampler lives in `triskel_mc.runner.run_ct_mcmc`. It follows the
continuous-time product-space design implemented in the code:

* **Birth/death clocks.** Each chain/walker pair keeps its own exponential
  clock based on `compute_bd_hazards_all`, which evaluates the likelihood
  impact of toggling every slot in the product-space mask `m` and builds
  birth/death intensities.【F:src/triskel_mc/birth_death.py†L32-L150】 When a
  clock fires, the sampler updates the active slot mask and logs the event in
  `EventLog` (and optionally `RunTrace`).【F:src/triskel_mc/runner.py†L73-L193】
* **Global MH ticks.** A Poisson clock with rate `rho_mh` drives batched Gibbs
  Metropolis–Hastings updates over the currently active slots. Stretch,
  random-walk, differential-evolution, and parallel-tempering swap moves are
  applied in sequence to the active subspace via utilities in
  `triskel_mc.mh_moves`.【F:src/triskel_mc/runner.py†L194-L558】【F:src/triskel_mc/mh_moves.py†L1-L151】
* **Optional tracing.** When provided a `TraceConfig`, the runner snapshots
  selected temperature indices after each birth/death event and each MH
  submove, storing them in `RunTrace` for later inspection or debugging.【F:src/triskel_mc/states.py†L49-L140】【F:src/triskel_mc/runner.py†L46-L69】

The compatibility module `triskel_mc.bd_mh_step_numpy` re-exports the same API
under its legacy name for users migrating from the original monolithic
implementation.【F:src/triskel_mc/bd_mh_step_numpy.py†L1-L13】 The package root
also re-exports the runner, state containers, and helper functions for
convenient importing.【F:src/triskel_mc/__init__.py†L1-L8】

## Core API

* **`run_ct_mcmc`**: Execute a full continuous-time run given initial
  `PTState` (continuous parameters) and `PSState` (product-space mask and
  components), plus likelihood/prior callbacks and sampler hyperparameters.
* **`compute_bd_hazards_all`**: Vectorized helper that returns the birth/death
  log-hazards and total intensities for every chain/walker combination.
* **`make_batched_loglik_masked`**: Wrap a single-configuration JAX likelihood
  into a NumPy-friendly batched evaluator used throughout the sampler.
* **`redblue_mask_np`**, **`propose_*_slot_np`**: Proposal utilities for the
  Gibbs submoves, kept slot-aware to operate only on active coordinates.
* **`RunTrace`, `EventLog`, `PTState`, `PSState`**: Dataclasses in
  `triskel_mc.states` that structure inputs and optional outputs for the
  continuous-time workflow.
