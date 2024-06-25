### for 3 and 5 ###
using Plots
using JLD2, FileIO
using LaTeXStrings


l = @layout [a b; c d]

# make sure to have the results for all step sizes in the working directory beore producing the plots

# step size = 8e-4
@load "35_MH_stepsize8e-4_step10000.jld2" acc acc_time
p1 = plot(acc_time, acc, ylabel="Test Accuracy", xlims=(0, 2.5), ylims=(0.4, 1), xticks = range(0, 4, 9), yticks = range(0.3, 1, 8), title="step size=0.0008", label="", linewidth=2)
@load "35_TunaMH_stepsize8e-4_step100000.jld2" acc acc_time
plot!(p1, acc_time, acc, label="", linewidth=2)
@load "35_TunaMH_SGLD_stepsize8e-4_step100000.jld2" acc acc_time
plot!(p1, acc_time, acc, label="", linewidth=2)
@load "35_MALA_clip_stepsize8e-4_step10000.jld2" acc acc_time
plot!(p1, acc_time, acc, label="", linewidth=2)
@load "35_HMC_lf5_clip_stepsize8e-4_step10000.jld2" acc acc_time
plot!(p1, acc_time, acc, label="", linewidth=2, color = :orange)


# step size = 1e-3
@load "35_MH_stepsize1e-3_step10000.jld2" acc acc_time
p2 = plot(acc_time, acc, xlims=(0, 2.5), ylims=(0.4, 1), xticks = range(0, 4, 9), yticks = range(0.3, 1, 8), title="step size=0.001", label="", linewidth=2)
@load "35_TunaMH_stepsize1e-3_step100000.jld2" acc acc_time
plot!(p2, acc_time, acc, label="", linewidth=2)
@load "35_TunaMH_SGLD_stepsize1e-3_step100000.jld2" acc acc_time
plot!(p2, acc_time, acc, label="", linewidth=2)
@load "35_MALA_clip_stepsize1e-3_step10000.jld2" acc acc_time
plot!(p2, acc_time, acc, label="", linewidth=2)
@load "35_HMC_lf5_clip_stepsize1e-3_step10000.jld2" acc acc_time
plot!(p2, acc_time, acc, label="", linewidth=2, color = :orange)


# step size = 2e-3
@load "35_MH_stepsize2e-3_step10000.jld2" acc acc_time
p3 = plot(acc_time, acc, label="",  ylabel="Test Accuracy", xlims=(0, 2.5), ylims=(0.4, 1), xticks = range(0, 4, 9), yticks = range(0.3, 1, 8), title="step size=0.002", linewidth=2)
@load "35_TunaMH_stepsize2e-3_step100000.jld2" acc acc_time
plot!(p3, acc_time, acc, label="", linewidth=2)
@load "35_TunaMH_SGLD_stepsize2e-3_step100000.jld2" acc acc_time
plot!(p3, acc_time, acc, label="", linewidth=2)
@load "35_MALA_clip_stepsize2e-3_step10000.jld2" acc acc_time
plot!(p3, acc_time, acc, label="", linewidth=2)
@load "35_HMC_lf5_clip_stepsize2e-3_step10000.jld2" acc acc_time
plot!(p3, acc_time, acc, label="", linewidth=2, color = :orange)


# step size = 4e-3
@load "35_MH_stepsize4e-3_step10000.jld2" acc acc_time
p4 = plot(acc_time, acc, label="MH", xlims=(0, 2.5), ylims=(0.4, 1), xticks = range(0, 4, 9), yticks = range(0.3, 1, 8), title="step size=0.004", linewidth=2)
@load "35_TunaMH_stepsize4e-3_step100000.jld2" acc acc_time
plot!(p4, acc_time, acc, label="TunaMH", linewidth=2)
@load "35_TunaMH_SGLD_stepsize4e-3_step100000.jld2" acc acc_time
plot!(p4, acc_time, acc, label="TunaMH-SGLD", linewidth=2)
@load "35_MALA_clip_stepsize4e-3_step10000.jld2" acc acc_time
plot!(p4, acc_time, acc, label="MALA", linewidth=2)
@load "35_HMC_lf5_clip_stepsize4e-3_step10000.jld2" acc acc_time
plot!(p4, acc_time, acc, label="HMC-5", linewidth=2, color = :orange)


p_time = plot(p1, p2, p3, p4, layout=l, xlabel="Time(s)", plot_title="3 and 5: Test Accuracy vs Time", size=(800, 600), dpi=1000, legend=:bottomright)
