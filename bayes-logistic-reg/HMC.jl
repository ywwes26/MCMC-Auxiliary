using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using MLDatasets
using MultivariateStats
using JLD2, FileIO


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--pca_dim"
            help = "pca dim"
            arg_type = Int64
            default = 50
        "--stepsize"
            help = "stepsize"
            arg_type = Float64
            default = 5e-2
        "--leapfrog"
            help = "number of leapfrog steps"
            arg_type = Int64
            default = 5
        "--T"
            help = "temperature"
            arg_type = Float64
            default = 1.0
        "--nsamples"
            help = "number of samples"
            arg_type = Int64
            default = 10000
        "--burnin"
            help = "number of samples as burnin"
            arg_type = Int64
            default = 0
    end

    return parse_args(s)
end


function main() 
    args = parse_commandline()
    train_x, train_y, test_x, test_y = generate_data(args)
    samples, avg_accept_prob, acc, acc_time, datause = run_sampler(args,train_x,train_y,test_x,test_y)
    @save "35_HMC_lf5_clip_stepsize5e-2_step10000.jld2" samples avg_accept_prob acc acc_time datause
    println("accuracy: $(acc[end])")
    println("runtime: $(acc_time[end])")
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
    sampler = HMC(X, y, args["leapfrog"], args["stepsize"], args["pca_dim"], args["T"])
    samples, avg_accept_prob, acc, acc_time, datause = mh_train(sampler, X, y, args["nsamples"], args["stepsize"], args["pca_dim"], args["T"], args["burnin"], test_x, test_y)
    return samples, avg_accept_prob, acc, acc_time, datause
end

struct HMC
    X::Array{Float64,2}
    y::Array{Int64,1}
    leapfrog::Int64
    stepsize::Float64
    data_size::Int64
    pca_dim::Int64
    T::Float64
    theta_prime::Array{Float64,1}
end

function mh_train(sampler::HMC, X::Array{Float64,2}, y::Array{Int64,1}, nsamples::Int64, stepsize::Float64, pca_dim::Int64, T::Float64, burnin::Int64, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    theta = kaiming_unif_init(pca_dim)
    acc_init = test(repeat(theta, 1, 2), test_x, test_y)
    #theta = zeros(pca_dim)
    succ = 0.
    samples = zeros(pca_dim, nsamples)
    iters = nsamples+burnin
    interval = 100
    K = Int(iters/interval)
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    k = 1
    total_runtime = 0.
    for i = 1:iters
        runtime = @elapsed begin
            (theta, sig) = next(sampler, theta)
            succ += sig
            if i > burnin
                for j = 1:pca_dim
                    samples[j,i-burnin] = theta[j]
                end
            end
        end
        total_runtime += runtime
        if (i % interval == 0)
            acc[k] = test(samples[:,1:i],test_x,test_y)
            acc_time[k] = total_runtime
            datause[k] = i*sampler.data_size
            k += 1
        end
    end
    avg_accept_prob = float(succ) / iters
    # add initial accuracy and time
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]
    return samples, avg_accept_prob, acc, acc_time, datause
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

function HMC(X::Array{Float64,2}, y::Array{Int64,1}, leapfrog::Int64, stepsize::Float64, pca_dim::Int64, T::Float64)
    theta_prime = zeros(pca_dim)
    return HMC(X, y, leapfrog, stepsize, size(X, 2), pca_dim, T, theta_prime);
end

function next(self::HMC, theta::Array{Float64,1})
    data_size = self.data_size
    sig = 0
    theta_prime, logp, logp_prime = proposal(self, theta)
    logmh = 0.0
    for i = 1:data_size
        (ll_old, ll_new) = logtarget(self, theta, theta_prime, i)
        logmh += ll_new - ll_old
    end
    # log proposal ratio
    logmh += logp_prime - logp
    acc_prob = stand_mh(self, logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return (theta, sig)
end


function grad_U(self::HMC, theta::Array{Float64,1})
    grad = zeros(self.pca_dim)
    for i = 1:self.data_size
        Xi_dot_theta = 0.0
        for j = 1:length(theta)
            Xi_dot_theta += self.X[j,i] * theta[j]
        end
        predict = sigmoid(Xi_dot_theta)
        yi = self.y[i]
        grad .+= self.X[:,i] * (predict - yi) / self.T
    end
    return grad
end

function leapfrog(self::HMC, theta::Array{Float64,1}, p::Array{Float64,1})
    # L leapfrog steps
    theta_prime = theta
    p_prime = p
    #p_prime .-= 0.5 * self.stepsize * grad_U(self, theta) # not clipping
    grad_theta = grad_U(self, theta)
    p_prime .-= 0.5 * self.stepsize * 2*grad_theta/norm(grad_theta)
    for i = 1:self.leapfrog
        theta_prime .+= self.stepsize*p_prime
        if i != self.leapfrog
            grad_prime1 = grad_U(self, theta_prime)
            p_prime .-= self.stepsize * 2*grad_prime1/norm(grad_prime1)
            #p_prime .-= self.stepsize*grad_U(self, theta_prime) # not clipping
        end
    end
    grad_prime2 = grad_U(self, theta_prime)
    p_prime .-= 0.5 * self.stepsize * 2*grad_prime2/norm(grad_prime2)
    #p_prime .-= 0.5*self.stepsize*grad_U(self, theta_prime) # not clipping
    p_prime = -p_prime
    return theta_prime, p_prime
end


function proposal(self::HMC, theta::Array{Float64,1})
    p = randn(self.pca_dim)
    theta_prime, p_prime = leapfrog(self, theta, p)
    self.theta_prime .= theta_prime
    logp = -0.5*dot(p, p)
    logp_prime = -0.5*dot(p_prime, p_prime)
    return self.theta_prime, logp, logp_prime
end


function stand_mh(self::HMC, u::Float64)
    return exp(u)
end

function sigmoid(z::Real) 
    return one(z) / (one(z) + exp(-z))
end

function logH(predict::Float64, y::Int64) 
    return y*log(predict) + (1-y)*log(1-predict)
end

function logtarget(self::HMC,theta::Array{Float64,1},theta_prime::Array{Float64,1},idx::Int64)
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
    return (logl, logl_prime)
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
