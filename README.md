# MCMC-Auxiliary
Collection of codes to replicate the simulation results in the paper: [Markov Chain Monte Carlo without evaluating the target: an auxiliary variable approach](https://arxiv.org/abs/2406.05242)

## Heterogeneous truncated Gaussian
Folder `hetero-trunc-gaussian` contains the source codes to replicate the simulation results in Section 5.1. Executing the file `TruncGaussian.jl` will output a jld2 file containing the samples, time per step and number of accepts for all MCMC algorithms. Parameters such as the target accept rate and step sizes can be changed manually inside this file.
