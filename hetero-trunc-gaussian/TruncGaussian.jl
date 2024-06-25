using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using MLDatasets
using MultivariateStats
using JLD2, FileIO


include("AliasSampler.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--data_size"
            help = "number of data points"
            arg_type = Int64
            default = 100000
        "--dim"
            help = "dimension of data"
            arg_type = Int64
            default = 20
        "--lam_const"
            help = "coefficient for lambda"
            arg_type = Float64
            default = 0.0005
        "--beta"
            help = "inverse temperature"
            arg_type = Float64
            default = 1e-5
    end
    return parse_args(s)
end


struct Params
    y::Array{Float64, 2}
    prec_y::Array{Float64, 2}
    data_size::Int64
    dim::Int64
    beta::Float64
    M::Array{Float64, 1}
    L::Float64
    lam::Float64
    rho::Array{Float64, 1}
    rho_sum::Float64
    Alias::AliasSampler
    stepsize::Float64
    nsamples::Int64
end


function generate_data(args::Dict)
    args = parse_commandline()
    cov_y = collect(Diagonal(LinRange(1, 0.05, args["dim"])))
    y = rand(MvNormal(zeros(args["dim"]), cov_y), args["data_size"])
    return y, cov_y
end


function Params(y::Array{Float64, 2}, prec_y::Array{Float64, 2}, dim::Int64, beta::Float64, lam_const::Float64, stepsize::Float64, nsamples::Int64)
    data_size = size(y)[2]
    lam_max = maximum(eigen(prec_y).values)
    M = vec(0.5 * beta * lam_max * sum((abs.(y) .+ 3.0).^2, dims=1))
    L = sum(M)
    lam = lam_const * (L^2)
    rho = lam * M ./ L .+ M
    rho_sum = sum(rho)
    weights = Weights(rho ./ rho_sum)
    Alias = AliasSampler(weights)
    return Params(y, prec_y, data_size, dim, beta, M, L, lam, rho, rho_sum, Alias, stepsize, nsamples)
end


function barker_proposal(self::Params, theta_cur::Array{Float64, 1}, grad::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        eta = randn() * self.stepsize
        p = 1.0 / (1.0 + exp(-eta * grad[j]))
        theta_prime[j] = theta_cur[j] + 2.0 * (rand(Binomial(1, p)) - 0.5) * eta
    end
    return theta_prime
end


function mala_proposal(self::Params, theta_cur::Array{Float64, 1}, grad::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + self.stepsize^2 * grad[j] + sqrt(2*self.stepsize^2) * randn()
    end
    return theta_prime
end


function rw_proposal(self::Params, theta_cur::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + self.stepsize * randn()
    end
    return theta_prime
end


# get phi_i needed for PoissonMH truncated gaussian
function get_phi(theta::Array{Float64,1}, y_i::Array{Float64,1}, beta::Float64, prec_y::Array{Float64,2}, M_i::Float64)
    phi_i = -0.5 * beta * (theta - y_i)' * prec_y * (theta - y_i) + M_i
    return phi_i
end


# gradient of \log pi(\theta \mid x) P_{\theta}(\omega_1))
function grad_barker(theta::Array{Float64,1}, y_i::Array{Float64,1}, s_i::Int64, phi_i::Float64, lam::Float64, beta::Float64, prec_y::Array{Float64,2}, M_i::Float64, L::Float64)
    grad = -beta * prec_y * (theta - y_i) * s_i / (lam * M_i / L + phi_i)
    return grad
end


# gradient of \log pi(\theta \mid x)
function grad_mala(theta::Array{Float64,1}, y_i::Array{Float64,1}, beta::Float64, prec_y::Array{Float64,2})
    grad = -beta * prec_y * (theta .- y_i)
    return grad
end


# leapfrog step for hmc
function leapfrog(self::Params, theta::Array{Float64,1}, p::Array{Float64,1}, grad::Array{Float64,1})
    p += self.stepsize * 0.5 * grad
    theta += self.stepsize * p
    p += self.stepsize * 0.5 * grad
    return theta, p
end


# quick sampling of poisson given the sum of their parameters
function quick_poisson(self::Params, theta_cur::Array{Float64,1})
    s = zeros(Int64, self.data_size)
    B = rand(Poisson(self.rho_sum))
    for b=1:B
        idx = rand(self.Alias)
        M_idx = self.M[idx]
        phi_idx = get_phi(theta_cur, self.y[:,idx], self.beta, self.prec_y, M_idx)
        add_prob = (self.lam * M_idx + self.L * phi_idx) / (self.lam * M_idx + self.L * M_idx)
        if rand() <= add_prob
            s[idx] += 1
        end
    end
    return s
end


# poissonmh
function poismh(self::Params, theta_init::Array{Float64, 1})
    y = self.y
    M = self.M
    L = self.L
    lam = self.lam
    beta = self.beta
    prec_y = self.prec_y
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # time per step
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("poismh step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if any(x -> x >= 3 || x <= -3, theta_prime)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            s = quick_poisson(self, theta_cur)
            I = findall(x -> x > 0, s)
            bs = length(I)
            phi_cur = [get_phi(theta_cur, y[:, i], beta, prec_y, M[i]) for i in I]
            phi_prime = [get_phi(theta_prime, y[:, i], beta, prec_y, M[i]) for i in I]
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            if log(rand()) <= logp_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end


# poisson-barker
function pois_barker(self::Params, theta_init::Array{Float64,1})
    y = self.y
    M = self.M
    L = self.L
    lam = self.lam
    beta = self.beta
    prec_y = self.prec_y
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # time per step
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("barker step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                bs = length(I)
                phi_cur = [get_phi(theta_cur, y[:, i], beta, prec_y, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[:, i], s[i], phi_cur[idx_phi], lam, beta, prec_y, M[i], L)
                end
                theta_prime = barker_proposal(self, theta_cur, grad_cur)
            end
            if any(x -> x >= 3 || x <= -3, theta_prime)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[:, i], beta, prec_y, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[:, i], s[i], phi_prime[idx_phi_prime], lam, beta, prec_y, M[i], L)
            end
            # log proposal ratio
            diff = theta_prime .- theta_cur
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            log_prop_ratio = sum(log.(1.0 .+ exp.(-diff .* grad_cur)) .- log.(1.0 .+ exp.(diff .* grad_prime)))
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end


# poisson-mala
function pois_mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    M = self.M
    L = self.L
    lam = self.lam
    beta = self.beta
    prec_y = self.prec_y
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # time per step
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("poismh mala step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                bs = length(I)
                phi_cur = [get_phi(theta_cur, y[:, i], beta, prec_y, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[:, i], s[i], phi_cur[idx_phi], lam, beta, prec_y, M[i], L)
                end
                theta_prime = mala_proposal(self, theta_cur, grad_cur)
            end
            if any(x -> x >= 3 || x <= -3, theta_prime)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[:, i], beta, prec_y, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[:, i], s[i], phi_prime[idx_phi_prime], lam, beta, prec_y, M[i], L)
            end
            # log proposal ratio
            diff = theta_prime .- theta_cur
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            log_prop_ratio = -0.5*(norm(-diff .- self.stepsize^2*grad_prime)^2 - norm(diff .-self.stepsize^2*grad_cur)^2)/(2*self.stepsize^2)
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end


# standard mh
function mh(self::Params, theta_init::Array{Float64,1})
    y = self.y
    M = self.M
    beta = self.beta
    prec_y = self.prec_y
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # time per step
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("mh step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if any(x -> x >= 3 || x <= -3, theta_prime)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[:, i], beta, prec_y, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[:, i], beta, prec_y, M[i]) for i in 1:self.data_size]
            logp_ratio = sum(phi_prime .- phi)
            if log(rand()) <= logp_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end



# MALA
function mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    M = self.M
    beta = self.beta
    prec_y = self.prec_y
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # time per step
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("mala step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                grad = zeros(self.dim)
                for i in 1:self.data_size
                    grad .+= grad_mala(theta_cur, y[:, i], beta, prec_y)
                end
                theta_prime = mala_proposal(self, theta_cur, grad)
            end
            if any(x -> x >= 3 || x <= -3, theta_prime)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[:, i], beta, prec_y, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[:, i], beta, prec_y, M[i]) for i in 1:self.data_size]
            grad_prime = zeros(self.dim)
            for i in 1:self.data_size
                grad_prime .+= grad_mala(theta_prime, y[:, i], beta, prec_y)
            end
            logp_ratio = sum(phi_prime .- phi)
            diff = theta_prime .- theta_cur
            log_prop_ratio = -0.5*(norm(-diff .- self.stepsize^2*grad_prime)^2 - norm(diff .- self.stepsize^2*grad)^2)/(2*self.stepsize^2)
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end


function run_sampler(theta_init::Array{Float64,1}, stepsize::Array{Float64,1}, steps::Array{Int,1})
    args = parse_commandline()
    y, cov_y = generate_data(args)
    prec_y = inv(cov_y)
    lam_max = maximum(eigen(prec_y).values)
    dim = size(prec_y)[1]
    params_mh = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[1], steps[1])
    params_mala = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[2], steps[2])
    params_poismh = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[3], steps[3])
    params_poisbarker = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[4], steps[4])
    params_poismala = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[5], steps[5])
    params_hmc = Params(y, prec_y, dim, args["beta"], args["lam_const"], stepsize[6], steps[6])

    # a large step of rejection sampling to estimate true theta
    theta_true_untrunc = rand(MvNormal(vec(mean(y, dims=2)), cov_y./(size(y)[2]*args["beta"])), 1000000)
    theta_true = []
    for i in 1:1000000
        if all(-3 .<= theta_true_untrunc[:, i] .<= 3)
            push!(theta_true, theta_true_untrunc[:, i])
        end
    end
    # all samplers
    mh_samples, accept_mh, time_mh = mh(params_mh, theta_init)
    mala_samples, accept_mala, time_mala = mala(params_mala, theta_init)
    poismh_samples, accept_poismh, time_poismh = poismh(params_poismh, theta_init)
    poisbarker_samples, accept_poisbarker, time_poisbarker = poismh_barker(params_poisbarker, theta_init)
    poismala_samples, accept_poismala, time_poismala = poismh_mala(params_poismala, theta_init)
    return mh_samples, mala_samples, poismh_samples, poisbarker_samples, poismala_samples, theta_true, time_mh, time_mala, time_poismh, time_poisbarker, time_poismala, accept_mh, accept_mala, accept_poismh, accept_poisbarker, accept_poismala
end


args = parse_commandline()
theta_init = randn(args["dim"])


# set step sizes for each algorithm
target_rate = 0.25
# mh, mala, poismh, poisbarker, poismala
if (target_rate == 0.25)
    stepsize_list = [0.29, 0.36, 0.3, 0.6, 0.365]
end
if (target_rate == 0.4)
    stepsize_list = [0.2, 0.31, 0.2, 0.5, 0.315]
end
if (target_rate == 0.55)
    stepsize_list = [0.15, 0.27, 0.14, 0.4, 0.275]
end

steps = [5000, 2500, 50000, 20000, 20000, 2000]

mh_samples, mala_samples, poismh_samples, poisbarker_samples, poismala_samples, theta_true, time_mh, time_mala, time_poismh, time_poisbarker, time_poismala, accept_mh, accept_mala, accept_poismh, accept_poisbarker, accept_poismala = run_sampler(theta_init, stepsize_list, steps)


# save results
function save_params_to_file()
    args = parse_commandline()
    dim = args["dim"]
    data_size = args["data_size"]
    beta = args["beta"]
    lam_const =  args["lam_const"]
    mean_true = mean(theta_true, dims=1)[1]
    cov_true = diag(cov(theta_true))
    filename = """TruncGaussian-dim$(dim)-data_size$(data_size)-lam_const$(lam_const)-target_rate$(target_rate)-beta$(beta)-round$(round)""" * ".jld2"
    @save filename theta_init mean_true cov_true mh_samples mala_samples poismh_samples poisbarker_samples poismala_samples time_mh time_mala time_poismh time_poisbarker time_poismala accept_mh accept_mala accept_poismh accept_poisbarker accept_poismala
end
save_params_to_file()

