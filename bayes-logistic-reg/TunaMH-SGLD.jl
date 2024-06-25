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
        "--pca_dim"
            help = "pca dim"
            arg_type = Int64
            default = 50
        "--grad_size"
            help = "grad size"
            arg_type = Int64
            default = 20
        "--stepsize"
            help = "stepsize"
            arg_type = Float64
            default = 4e-3
        "--T"
            help = "temperature"
            arg_type = Float64
            default = 1.0
        "--nsamples"
            help = "number of samples"
            arg_type = Int64
            default = 100000
        "--burnin"
            help = "number of samples as burnin"
            arg_type = Int64
            default = 0
        "--lam"
            help = "lambda to control minibatch size"
            arg_type = Float64
            default = 1e-5
    end
    return parse_args(s)
end


function main() 
    args = parse_commandline()
    train_x, train_y, test_x, test_y = generate_data(args)
    samples, total_bs, avg_accept_prob, acc, acc_time, datause = run_sampler(args,train_x,train_y,test_x,test_y)
    @save "35_TunaMH_SGLD_stepsize4e-3_step100000.jld2" samples total_bs avg_accept_prob acc acc_time datause
    println("accuracy: $(acc[end])")
    println("runtime: $(acc_time[end])")
    println("Avg Batch Size: $(total_bs/(args["burnin"]+args["nsamples"]))")
    println("Avg Acceptance Prob: $(avg_accept_prob)")
end


function generate_data(args::Dict)
    Random.seed!(2024)
    train_x, train_y = MNIST(:train)[:]
    train_x = reshape(train_x, 784, :)
    idx1 = findall(x -> x == 3, train_y)
    idx2 = findall(x -> x == 5, train_y)
    idx = sort(vcat(idx1, idx2))
    train_y[idx1] .= 0
    train_y[idx2] .= 1
    train_y = train_y[idx]
    train_x = train_x[:,idx]

    test_x, test_y = MNIST(:test)[:]
    test_x = reshape(test_x, 784, :)
    idx1 = findall(x -> x == 3, test_y)
    idx2 = findall(x -> x == 5, test_y)
    idx = sort(vcat(idx1, idx2))
    test_y[idx1] .= 0
    test_y[idx2] .= 1
    test_y = test_y[idx]
    test_x = test_x[:,idx]
    train_x = convert(Array{Float64},train_x)
    test_x = convert(Array{Float64},test_x)
    M = fit(PCA, train_x; maxoutdim=args["pca_dim"])
    train_x = transform(M, train_x)
    test_x = transform(M, test_x)
    return (train_x, train_y, test_x, test_y)
end


function run_sampler(args::Dict, X::Array{Float64,2}, y::Array{Int64,1}, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    sampler = SGLD(X, y, args["grad_size"], args["stepsize"], args["pca_dim"], args["lam"], args["T"])
    samples, total_bs, avg_accept_prob, acc, acc_time, datause = new_poissonmh_train(sampler, X, y, args["nsamples"], args["stepsize"], args["pca_dim"], args["lam"], args["T"], args["burnin"], test_x, test_y)
    return samples, total_bs, avg_accept_prob, acc, acc_time, datause
end


struct SGLD
    X::Array{Float64,2}
    y::Array{Int64,1}
    grad_size::Int64
    stepsize::Float64
    data_size::Int64
    pca_dim::Int64
    c1::Float64
    psi::Array{Float64,1}
    Psi::Float64
    gamma_A::AliasSampler
    lam::Float64
    T::Float64
    theta_prime::Array{Float64,1}
end


function new_poissonmh_train(sampler::SGLD, X::Array{Float64,2}, y::Array{Int64,1}, nsamples::Int64, stepsize::Float64, pca_dim::Int64, lam::Float64, T::Float64, burnin::Int64, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    theta = kaiming_unif_init(pca_dim)
    acc_init = test(repeat(theta, 1, 2), test_x, test_y)
    succ = 0.
    samples = zeros(pca_dim, nsamples)
    total_bs = 0.
    iters = nsamples+burnin
    interval = 100
    K = Int(nsamples/interval)
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    k = 1
    total_runtime = 0.
    for i = 1:iters
        runtime = @elapsed begin
            (theta, sig, bs) = next(sampler, theta)
            total_bs += bs
            succ += sig
            if i > burnin
                for j = 1:pca_dim
                    samples[j,i-burnin] = theta[j]
                end
            end
        end
        total_runtime += runtime
        if (i>burnin && i % interval == 0)
            acc[k] = test(samples[:,1:i-burnin],test_x,test_y)
            acc_time[k] = total_runtime
            datause[k] = total_bs
            k += 1
        end
    end
    avg_accept_prob = float(succ) / iters
    # add initial accuracy and time
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]
    return samples, total_bs, avg_accept_prob, acc, acc_time, datause
end


function kaiming_unif_init(pca_dim::Int64)
    a = sqrt(5.0)
    fan = pca_dim
    gain = sqrt(2.0 / (1 + a^2))
    std = gain / sqrt(fan)
    bound = sqrt(3.0) * std
    theta = 2*bound*rand(pca_dim).-bound
    return theta
end


function SGLD(X::Array{Float64,2}, y::Array{Int64,1}, grad_size::Int64, stepsize::Float64, pca_dim::Int64, lam::Float64, T::Float64)
    c1 = 1.0
    psi = c1 * sqrt.(vec(sum(X.^2; dims=1)))/T
    Psi = sum(psi)
    gamma = Weights(psi ./ Psi)
    gamma_A = AliasSampler(gamma)
    theta_prime = zeros(pca_dim)
    return SGLD(X, y, grad_size, stepsize, size(X, 2), pca_dim, c1, psi, Psi, gamma_A, lam, T, theta_prime);
end


function next(self::SGLD, theta::Array{Float64,1})
    data_size = self.data_size
    sig = 0
    theta_prime, grad_idx, proposal_mean, proposal_mean_prime = proposal(self, theta)
    diff_norm = dist(theta_prime, theta)
    L = diff_norm * self.Psi
    lam = self.lam*L^2
    N = lam + L
    s = rand(Poisson(N))
    bs = 0
    logmh = 0.0
    for ii = 1:s
        idx = rand(self.gamma_A)
        M_i = diff_norm * self.psi[idx]
        (phi_old, phi_new) = get_phi_i(self, idx, M_i, theta, theta_prime)
        ps = (lam*M_i + L*phi_old) / (lam*M_i + L*M_i)
        if (rand() <= ps)
            bs += 1
            logmh += log(1 + L/(lam*M_i)*phi_new) - log(1 + L/(lam*M_i)*phi_old)
        end
    end
    # log proposal ratio
    logmh += logp_proposal(self, theta, proposal_mean_prime) - logp_proposal(self, theta_prime, proposal_mean)
    acc_prob = stand_mh(self, logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return (theta, sig, s)
end


function dist(x::Array{Float64,1}, y::Array{Float64,1})
    @assert(length(x) == length(y))
    acc = 0.0;
    for i = 1:length(x)
        acc += (x[i] - y[i])^2
    end
    return sqrt(acc)
end


function proposal(self::SGLD, theta::Array{Float64,1})
    grad_idx = sample(1:self.data_size, self.grad_size, replace=false)
    X_grad_idx = self.X[:,grad_idx]
    y_grad_idx = self.y[grad_idx]
    sigmoid_term = sigmoid.(X_grad_idx' * theta) .- y_grad_idx
    grad = X_grad_idx * sigmoid_term / self.T
    # clip grad norm to 2
    proposal_mean = theta - 0.5self.stepsize^2*self.data_size/self.grad_size * 2grad/norm(grad)
    self.theta_prime .= proposal_mean .+ self.stepsize .* randn(self.pca_dim)
    # proposal
    sigmoid_term_prime = sigmoid.(X_grad_idx' * self.theta_prime) .- y_grad_idx
    grad_prime = X_grad_idx * sigmoid_term_prime / self.T
    # clip grad norm to 2
    proposal_mean_prime = self.theta_prime - 0.5self.stepsize^2*self.data_size/self.grad_size * 2grad_prime/norm(grad_prime)
    return self.theta_prime, grad_idx, proposal_mean, proposal_mean_prime
end


function grad_U_i(x::Array{Float64, 1}, sigmoid_term::Array{Float64, 1})
    return sum(x .* sigmoid_term)
end


function stand_mh(self::SGLD, u::Float64)
    return exp(u)
end


function sigmoid(z::Real) 
    return one(z) / (one(z) + exp(-z))
end


function logH(predict::Float64, y::Int64) 
    return y*log(predict) + (1-y)*log(1-predict)
end


function logp_proposal(self::SGLD, theta::Array{Float64, 1}, prop_mean::Array{Float64, 1})
    return -0.5sum((theta .- prop_mean).^2)/(self.stepsize^2)
end


function get_phi_i(self::SGLD,idx::Int64,Mi::Float64,theta::Array{Float64,1},theta_prime::Array{Float64,1})
    Xi_dot_theta = 0.0;
    Xi_dot_theta_prime = 0.0;
    for j = 1:length(theta)
        Xi_dot_theta += self.X[j,idx] * theta[j]
        Xi_dot_theta_prime += self.X[j,idx] * theta_prime[j]
    end
    predict = sigmoid(Xi_dot_theta)
    predict_prime = sigmoid(Xi_dot_theta_prime)
    yi = self.y[idx]
    logl = logH(predict, yi) / self.T
    logl_prime = logH(predict_prime, yi) / self.T
    return (0.5*(logl - logl_prime + Mi), 0.5*(logl_prime - logl + Mi))
end


function test(samples::Array{Float64,2}, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    avg_sample = mean(samples, dims=2)
    N = size(test_x, 2)
    acc = 0.0
    for i = 1:N
        predict = dot(avg_sample, test_x[:,i])
        if predict > 0 
            if test_y[i] == 1
                acc += 1.0
            end
        else
            if test_y[i] == 0
                acc += 1.0
            end
        end
    end
    return acc/N
end


main()

