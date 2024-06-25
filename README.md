# MCMC-Auxiliary
Collection of codes to replicate the simulation results in the paper: [Markov Chain Monte Carlo without evaluating the target: an auxiliary variable approach](https://arxiv.org/abs/2406.05242)

## Heterogeneous truncated Gaussian
Folder `hetero-trunc-gaussian` contains the source codes to replicate the simulation results in Section 5.1. Executing the file `TruncGaussian.jl` will output a jld2 file containing the samples, time per step and number of accepts for all MCMC algorithms. Parameters such as the target accept rate and step sizes can be changed manually inside this file.

## Robust linear regression
Folder `robust-lin-reg` contains the source codes to replicate the simulation results in Section 5.2. Executing the file `RobustLinReg.jl` will output a jld2 file containing the samples, time per step and number of accepts for all MCMC algorithms. To replicate the Figure 3 in the paper, see `plot-RobustLinReg.jl` for details.

## Bayesian logistic regression on MNIST
Folder `bayes-logistic-reg` contains the source codes to replicate the simulation results in Section 5.3. Each algorithm is implemented in separate files labeled by their respective names. For example, executing `TunaMH-SGLD.jl` will output a file containing the simulation results for TunaMH-SGLD. To reproduce Figure 5 in the paper, see `plot-mnist.jl` for details.

## Acknowledgement
The implementaion in this repository is based on the previous implementation of PoissonMH and TunaMH, available at https://github.com/ruqizhang/tunamh.
