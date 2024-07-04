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
            default = 10
        "--c"
            help = "bound on 2-norm"
            arg_type = Float64
            default = 15.0
        "--df"
            help = "degrees of freedom"
            arg_type = Int64
            default = 4
        "--lam_const"
            help = "coefficient for lambda=lam_const*L^2"
            arg_type = Float64
            default = 0.01
        "--beta"
            help = "inverse temperature"
            arg_type = Float64
            default = 1e-4
    end
    return parse_args(s)
end


struct Params
    y::Array{Float64, 1}
    x::Array{Float64, 2}
    data_size::Int64
    dim::Int64
    c::Float64
    df::Int64
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
    #=
    # correlated x
    cov_x = zeros(args["dim"], args["dim"])
    for i=1:args["dim"]
        for j=1:args["dim"]
            cov_x[i, j] = 0.5^abs(i-j)
        end
    end
    x = rand(MvNormal(zeros(args["dim"]), cov_x), args["data_size"])
    =#
    x = randn(args["dim"], args["data_size"])
    theta_true = ones(args["dim"])
    y = vec(x' * theta_true .+ randn(args["data_size"]))
    return y, x, theta_true
end


function Params(y::Array{Float64, 1}, x::Array{Float64, 2}, dim::Int64, c::Float64, df::Int64, beta::Float64, lam_const::Float64, stepsize::Float64, nsamples::Int64)
    data_size = size(y)[1]
    x_2norm = sqrt.(vec(sum(x.^2, dims=1)))
    y_abs = abs.(y)
    M = zeros(data_size)
    for i=1:data_size
        M[i] = beta * (df+1)/2 * log(1 + (1/df) * ((y_abs[i] + x_2norm[i] * c)^2))
    end
    L = sum(M)
    lam = lam_const * (L^2)
    rho = lam * M ./ L .+ M
    rho_sum = sum(rho)
    weights = Weights(rho ./ rho_sum)
    Alias = AliasSampler(weights)
    return Params(y, x, data_size, dim, c, df, beta, M, L, lam, rho, rho_sum, Alias, stepsize, nsamples)
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


function rw_proposal(self::Params, theta_cur::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + self.stepsize * randn()
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


# get phi_i needed for PoissonMH in robust linear regression
function get_phi(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, df::Int64, beta::Float64, M_i::Float64)
    phi_i = beta * -(df+1)/2 * log(1 + (1/df)*(y_i - theta' * x_i)^2) + M_i
    return phi_i
end


# gradient of \log pi(\theta \mid x) P_{\theta}(\omega_1))
function grad_barker(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, s_i::Int64, df::Int64, phi_i::Float64, lam::Float64, beta::Float64, M_i::Float64, L::Float64)
    err = y_i - theta' * x_i
    grad_phi = beta * -(df + 1)/2 * 1/(1 + err^2/df) * 2*err/df * -x_i
    grad = grad_phi * s_i / (lam * M_i / L + phi_i)
    return grad
end


# gradient of \log pi(\theta \mid x)
function grad_mala(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, df::Int64, beta::Float64)
    err = y_i - theta' * x_i
    grad = beta * -(df + 1)/2 * 1/(1 + err^2/df) * 2*err/df * -x_i
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
        phi_idx = get_phi(theta_cur, self.y[idx], self.x[:, idx], self.df, self.beta, M_idx)
        add_prob = (self.lam * M_idx + self.L * phi_idx) / (self.lam * M_idx + self.L * M_idx)
        if rand() <= add_prob
            s[idx] += 1
        end
    end
    return s
end


# PoissonMH
function poismh(self::Params, theta_init::Array{Float64, 1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("poismh step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            s = quick_poisson(self, theta_cur)
            I = findall(x -> x > 0, s)
            bs = length(I)
            phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
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


# Poisson-Barker
function pois_barker(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # per step time
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("poismh barker step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                bs = length(I)
                phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[i], x[:, i], s[i], df, phi_cur[idx_phi], lam, beta, M[i], L)
                end
                theta_prime = barker_proposal(self, theta_cur, grad_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[i], x[:, i], s[i], df, phi_prime[idx_phi_prime], lam, beta, M[i], L)
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


# Poisson_MALA
function pois_mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # per step time
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("poismh mala step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                bs = length(I)
                phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[i], x[:, i], s[i], df, phi_cur[idx_phi], lam, beta, M[i], L)
                end
                theta_prime = mala_proposal(self, theta_cur, grad_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[i], x[:, i], s[i], df, phi_prime[idx_phi_prime], lam, beta, M[i], L)
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


# standard Metropolis-Hastings
function mh(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # per step time
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("mh step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
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


# mala
function mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # per step time
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("mala step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                grad = zeros(self.dim)
                for i in 1:self.data_size
                    grad .+= grad_mala(theta_cur, y[i], x[:, i], df, beta)
                end
                theta_prime = mala_proposal(self, theta_cur, grad)
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            grad_prime = zeros(self.dim)
            for i in 1:self.data_size
                grad_prime .+= grad_mala(theta_prime, y[i], x[:, i], df, beta)
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



# HMC
function hmc(self::Params, theta_init::Array{Float64,1}, lf_step::Int64)
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    # per step time
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        println("hmc step: ", step)
        runtime = @elapsed begin
            prop_time = @elapsed begin
                # Initialize momentum from a standard normal distribution
                current_p = randn(self.dim)
                theta_prime = zeros(self.dim)
                theta_prime .= theta_cur
                p_prime = zeros(self.dim)
                p_prime .= current_p  
                # Perform leapfrog integration
                for _ in 1:lf_step
                    grad = zeros(self.dim)
                    for i in 1:self.data_size
                        grad .+= grad_mala(theta_prime, y[i], x[:, i], df, beta)
                    end
                    theta_prime, p_prime = leapfrog(self, theta_prime, p_prime, grad)
                end
                # Negate momentum at end of trajectory to make the proposal symmetric
                p_prime = -p_prime
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            logp_ratio = sum(phi_prime .- phi)
            log_prop_ratio = -0.5 * dot(p_prime, p_prime) + 0.5 * dot(current_p, current_p)
            # Accept or reject the proposal
            if log(rand()) < logp_ratio + log_prop_ratio
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


# run all samplers
function run_sampler(theta_init::Array{Float64,1}, stepsize::Array{Float64,1}, steps::Array{Int,1})
    args = parse_commandline()
    y, x, theta_true = generate_data(args)
    # set parameters
    params_mh = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[1], steps[1])
    params_mala = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[2], steps[2])
    params_poismh = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[3], steps[3])
    params_poisbarker = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[4], steps[4])
    params_poismala = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[5], steps[5])
    params_hmc = Params(y, x, args["dim"], args["c"], args["df"], args["beta"], args["lam_const"], stepsize[6], steps[6])

    # sampling
    mh_samples, accept_mh, time_mh = mh(params_mh, theta_init)
    mala_samples, accept_mala, time_mala = mala(params_mala, theta_init)
    poismh_samples, accept_poismh, time_poismh = poismh(params_poismh, theta_init)
    poisbarker_samples, accept_poisbarker, time_poisbarker = pois_barker(params_poisbarker, theta_init)
    poismala_samples, accept_poismala, time_poismala = pois_mala(params_poismala, theta_init)
    # hmc with 10 leapfrog steps
    hmc_samples, accept_hmc, time_hmc = hmc(params_hmc, theta_init, 10)
    return mh_samples, mala_samples, poismh_samples, poisbarker_samples, poismala_samples, hmc_samples, theta_true, time_mh, time_mala, time_poismh, time_poisbarker, time_poismala, time_hmc, accept_mh, accept_mala, accept_poismh, accept_poisbarker, accept_poismala, accept_hmc
end


# parse arguments and set initial theta
args = parse_commandline()
theta_init = randn(args["dim"])

# set step sizes for each algorithm
target_rate = 0.25
# mh, mala, pois, poisbarker, poismala, hmc
if (target_rate == 0.25)
    stepsize_list = [0.36, 0.488, 0.35, 0.68, 0.47, 0.074]
end
if (target_rate == 0.4)
    stepsize_list = [0.26, 0.42, 0.25, 0.55, 0.41, 0.064]
end
if (target_rate == 0.55)
    stepsize_list = [0.18, 0.38, 0.17, 0.465, 0.36, 0.052]
end


# set steps for each algorithm: mh, mala, poissonmh, poisson-barker, poisson-mala, hmc-10
steps = [4000, 4000, 40000, 20000, 20000, 2000]


# execute sampling
mh_samples, mala_samples, poismh_samples, poisbarker_samples, poismala_samples, hmc_samples, theta_true, time_mh, time_mala, time_poismh, time_poisbarker, time_poismala, time_hmc, accept_mh, accept_mala, accept_poismh, accept_poisbarker, accept_poismala, accept_hmc = run_sampler(theta_init, stepsize_list, steps)


# save results
function save_params_to_file()
    args = parse_commandline()
    data_size = args["data_size"]
    dim = args["dim"]
    c = args["c"]
    df = args["df"]
    lam_const = args["lam_const"]
    beta = args["beta"]
    filename = """RobustLinReg-dim$(dim)-data_size$(data_size)-c$(c)-df$(df)-lam_const$(lam_const)-beta$(beta)-target_rate$(target_rate)""" * ".jld2"
    @save filename theta_init mh_samples mala_samples poismh_samples poisbarker_samples poismala_samples hmc_samples theta_true time_mh time_mala time_poismh time_poisbarker time_poismala time_hmc accept_mh accept_mala accept_poismh accept_poisbarker accept_poismala accept_hmc
end
save_params_to_file()

