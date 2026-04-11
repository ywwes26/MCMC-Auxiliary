# Auxiliary-MCMC
Collection of code to reproduce the simulation results in the paper: [Markov Chain Monte Carlo without evaluating the target: an auxiliary variable approach](https://arxiv.org/abs/2406.05242)

## Install Julia and Dependencies

Download and install Julia v1.10.0 for your operating system from https://julialang.org/downloads/oldreleases/

Clone the GitHub repository:
```bash
git clone https://github.com/ywwes26/Auxiliary-MCMC.git
```

Change the directory to the repository and launch Julia:
```bash
cd Auxiliary-MCMC
julia --project=.
```

Install the required packages:
```julia
using Pkg
Pkg.instantiate()
```

## Run Experiments

The source code for each algorithm is provided in `src`, the scripts for running experiments are in `scripts`, and the code for computing ESS/s and KS statistics is in `eval`.

### 20-Dimensional Truncated Gaussian Example

The experiment can be run for a single round, e.g., with `target_rate=0.55`:

```bash
julia --project=. scripts/run_gaussian_20d.jl --target_rate 0.55 --round 1
```

Multiple rounds can be run using a for loop (this may be slow):

```bash
for rate in 0.25 0.4 0.55; do
  for r in $(seq 1 10); do
    julia --project=. scripts/run_gaussian_20d.jl \
      --target_rate $rate \
      --round $r
  done
done
```

The output (e.g., posterior sample trajectories, cumulative runtime trajectories, etc.) will be saved as `.jld2` files under `results/gaussian_20d` for evaluation.

If HPC resources are available, we recommend running the experiments using a SLURM job array (parallel and faster):

1. Create the run file `run.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=gaussian20d
#SBATCH --output=logs/gaussian_%A_%a.out
#SBATCH --error=logs/gaussian_%A_%a.err
#SBATCH --array=1-30
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module load julia

rates=(0.25 0.4 0.55)
R=10

idx=$((SLURM_ARRAY_TASK_ID - 1))
rate_idx=$((idx / R))
round=$((idx % R + 1))
rate=${rates[$rate_idx]}

julia --project=. scripts/run_gaussian_20d.jl \
  --target_rate $rate \
  --round $round
```

2. Create the log directory:

```bash
mkdir -p logs
```

3. Submit the jobs:
```bash
sbatch run.slurm
```

The number of steps for each algorithm can be customized:

```bash
julia --project=. scripts/run_gaussian_20d.jl \
  --target_rate 0.55 \
  --round 1 \
  --mh_steps 5000 \
  --mala_steps 2500 \
  --poismh_steps 50000 \
  --pois_barker_steps 20000 \
  --pois_mala_steps 20000 \
  --barker_steps 5000 \
  --sgld_steps 100000
```

### Robust Regression Example

Configurations for all three settings in the paper are provided:

| Experiment Name          | Dimension | Sample Size |
|------------------------|-----------|-------------|
| robust_reg_10d_n100000 | 10        | 100000      |
| robust_reg_50d_n100000 | 50        | 100000      |
| robust_reg_10d_n200000 | 10        | 200000      |

Run the experiment for configuration `robust_reg_10d_n100000` with `target_rate=0.55`:

```bash
julia --project=. scripts/run_robust_reg.jl \
  --experiment robust_reg_10d_n100000 \
  --target_rate 0.55 \
  --round 1
```

The output (e.g., posterior sample trajectories, cumulative runtime trajectories, etc.) will be saved as `.jld2` files under `results/robust_reg/` for evaluation.

Running multiple rounds and adjusting the number of steps for each algorithm are similar to the truncated Gaussian example.

### Bayesian Logistic Regression Example on MNIST

Run all algorithms for classifying 3's and 5's (or 7's and 9's using `--task mnist79`):

```bash
julia --project=. scripts/run_bayes_logistic_reg.jl --run_all --task mnist35
```

The number of steps (`nsamples`) and step sizes can be specified as follows:

```bash
julia --project=. scripts/run_bayes_logistic_reg.jl \
  --run_all \
  --task mnist35 \
  --pca_dim 50 \
  --burnin 0 \
  --mh_stepsize 4e-3 \
  --mh_nsamples 10000 \
  --mala_stepsize 4e-3 \
  --mala_nsamples 10000 \
  --hmc_stepsize 4e-3 \
  --hmc_nsamples 10000 \
  --tuna_mh_stepsize 4e-3 \
  --tuna_mh_nsamples 100000 \
  --tuna_sgld_stepsize 4e-3 \
  --tuna_sgld_nsamples 100000 \
  --barker_stepsize 4e-3 \
  --barker_nsamples 10000
```

To run a single algorithm:

```bash
julia --project=. scripts/run_bayes_logistic_reg.jl \
  --method tuna_sgld \
  --task mnist35 \
  --stepsize 4e-3 \
  --nsamples 100000
```

The output (e.g., posterior samples, test accuracy trajectories, runtime trajectories, etc.) will be saved as `.jld2` files under `results/bayes_logistic_reg/`, with filenames of the form  
`<task>_<method>_pca<pca_dim>_step<stepsize>_n<nsamples>_burn<burnin>_seed<seed>.jld2`.

## Evaluation

The scripts in `eval` automatically detect output files with different `target_rate` and `round` parameters in the `results` directory. For each `target_rate`, results from different rounds are aggregated to compute evaluation metrics.

Computing ESS/s for the truncated Gaussian example:
```bash
julia --project=. eval/ess_gaussian_20d.jl
```

Computing ESS/s for the robust regression example:
```bash
julia --project=. eval/ess_robust_reg.jl --experiment robust_reg_10d_n100000
```

Computing KS statistics for the truncated Gaussian example (`--save_theta_true` must be enabled when running the experiments in order to compute KS statistics. This option is disabled by default because saving the reference samples requires substantial disk space):
```bash
julia --project=. eval/ks_gaussian_20d.jl
```

## Acknowledgement

The implementation in this repository is based on the original implementations of PoissonMH and TunaMH, available at https://github.com/ruqizhang/tunamh.

