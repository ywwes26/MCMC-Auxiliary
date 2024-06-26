using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using MLDatasets
using MultivariateStats
using JLD2, FileIO
# change target rate accordingly to produce all plots
target_rate = 0.25
@load "RobustLinReg-dim10-data_size100000-c15.0-df4-lam_const0.01-beta0.0001-target_rate$(target_rate).jld2" theta_init mh_samples mala_samples poismh_samples poisbarker_samples poismala_samples hmc_samples theta_true time_mh time_mala time_poismh time_poisbarker time_poismala time_hmc accept_mh accept_mala accept_poismh accept_poisbarker accept_poismala accept_hmc


MSE_mh = zeros(4000)
MSE_mala = zeros(4000)
MSE_poismh = zeros(40000)
MSE_poisbarker = zeros(20000)
MSE_poismala = zeros(20000)
MSE_hmc = zeros(2000)


# get MSE for every step
for j in 1:4000
    MSE_mh[j] = mean((mean(mh_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end
for j in 1:4000
    MSE_mala[j] = mean((mean(mala_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end
for j in 1:40000
    MSE_poismh[j] = mean((mean(poismh_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end
for j in 1:20000
    MSE_poisbarker[j] = mean((mean(poisbarker_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end
for j in 1:20000
    MSE_poismala[j] = mean((mean(poismala_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end
for j in 1:2000
    MSE_hmc[j] = mean((mean(hmc_samples[:, 1:(j*1)], dims=2) .- theta_true).^2)
end



# If plotting 10 repeated simulations, make sure to average both time and MSE before using the code below
using PyCall
plt = pyimport("matplotlib.pyplot")
inset_loc = pyimport("mpl_toolkits.axes_grid1.inset_locator")
# pyplot
fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)
ax.plot(time_mh, MSE_mh, label="MH", color="dodgerblue", linewidth=2)
ax.plot(time_mala, MSE_mala, label="MALA", color="slategrey", linewidth=2)
ax.plot(time_poismh, MSE_poismh, label="PoissonMH", color="orangered", linewidth=2)
ax.plot(time_poisbarker, MSE_poisbarker, label="Poisson-Barker", color="mediumseagreen", linewidth=2)
ax.plot(time_poismala, MSE_poismala, label="Poisson-MALA", color="darkorange", linewidth=2)
ax.plot(time_hmc, MSE_hmc, label="HMC-10", color="purple", linewidth=2)
# set xlim
ax.set_xlim(0, 7)
ax.set_ylim(0, 1.5)
ax.set_xlabel("Time (s)", fontsize=26)
ax.set_ylabel("MSE", fontsize=28)
ax.tick_params(axis="both", labelsize=22)
ax.set_xticks(0:1:7)
ax.set_yticks(0:0.25:1.5)
ax.set_title("Accept Rate=0.25", fontsize=28)
ax.legend(fontsize=22, ncol=2)
ax.grid(alpha=0.4)

sub_axes = plt.axes([.52, .29, .35, .35])
sub_axes.plot(time_poismh, MSE_poismh, color="orangered", linewidth=2)
sub_axes.plot(time_poisbarker, MSE_poisbarker, color="mediumseagreen", linewidth=2)
sub_axes.plot(time_poismala, MSE_poismala, color="darkorange", linewidth=2)
sub_axes.set_xlim(0, 0.5)
sub_axes.set_ylim(0, 1)
sub_axes.tick_params(axis="both", labelsize=22)
sub_axes.set_xticks(0.1:0.1:0.5)
sub_axes.set_yticks(0.25:0.25:1)
sub_axes.grid(alpha=0.4)

inset_loc.mark_inset(ax, sub_axes, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()
