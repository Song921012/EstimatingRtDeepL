using CSV, DataFrames
using Plots
source_data = DataFrame(CSV.File("./DataSum.csv"))

scatter(1:137, source_data.case, xlabel = "Days after March 1", ylabel = "Daily Cases", label = "Real Data", ms = 5)
plot!(1:137, source_data.DeepLearningIt, label = "Deep Learning Method", w = 2)
plot!(1:137, source_data.EpiEstimIt, label = "EpiEstim Method", w = 2)
plot!(1:137, source_data.EpiNow2It, label = "EpiNow2 Method", w = 2)
plot!(1:137, source_data.EmsembleIt, label = "Ensemble Method", w = 2)

savefig("Figures/comparisonensembleIt.png")


plot(1:137, source_data.DeepLearningRt, label = "Deep Learning Method", w = 2)
plot!(1:137, source_data.EpiEstimRt, label = "EpiEstim Method", w = 2)
plot!(1:137, source_data.EpiNow2Rt, label = "EpiNow2 Method", w = 2)
plot!(1:137, source_data.EmsembleRt, label = "Ensemble Method", w = 2)
plot!(1:137, source_data.KalmanRt, label = "Kalman Method", w = 2)
xlabel!("Days after March 1")
ylabel!("Effective Reproduction Number")

savefig("Figures/comparisonensembleRt.png")