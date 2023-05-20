##
# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DataFrames
using CSV
using ComponentArrays
using OptimizationOptimisers
rng = Random.default_rng()
Random.seed!(14);

# Loading Data
source_data = DataFrame(CSV.File("./DeepLearningEffectiveReproductionNumber/Source_Data/Provincial_Daily_Totals.csv"))
data_on = source_data[source_data.Province.=="ONTARIO", :]
n = 30
m = 149
data_acc = data_on.TotalCases[(n+1):n+m+1]
data_daily = data_on.TotalCases[(n+1):n+m+1] - data_on.TotalCases[n:n+m]
display(plot(data_daily, label = "Daily Confirmed Cases", lw = 2))
display(plot(data_acc, label = "Accumulated Confirmed Cases", lw = 2))
data_daily[1]
println(length(data_acc))
trainingdata=Float32.(data_acc)
# set up neural differential equation models


using BSON: @load
@load "./DeepLearningEffectiveReproductionNumber/Saving_Data/ann_nn_ir.bason" ann
@load "./DeepLearningEffectiveReproductionNumber/Saving_Data/ann_para_irlbfgs.bason" psave
p, st = Lux.setup(rng, ann)
pinit = ComponentArray(p)
pfinal = ComponentArray(psave,getaxes(pinit))
function SIR_nn(du, u, p, t)
    I, H = u
    du[1] = 0.1 * min(5, abs(ann([t], p, st)[1][1])) * I - 0.1 * I
    du[2] = 0.1 * min(5, abs(ann([t], p, st)[1][1])) * I
end
u0 = Float32[1, data_acc[1]]
tspan = (0.0f0, 149.0f0)
tsteps = range(tspan[1], tspan[2], length=length(data_acc))
prob_neuralode = ODEProblem(SIR_nn, u0, tspan_data, ComponentArray(p_0))


# simulate the neural differential equation models
function predict_neuralode(θ)
    #Array(prob_neuralode(u0, p, st)[1])
    prob = remake(prob_neuralode, p=θ)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

predict_neuralode(pfinal)[2, :]
size(predict_neuralode(pfinal)[2, :]) == size(data_acc)


pred = predict_neuralode(pfinal)[2, :]
plt = scatter(tsteps, trainingdata, label="Accumulated cases")
plot!(plt, tsteps, pred, label="Predicted accumulated cases")
display(plot(plt))
savefig("./DeepLearningEffectiveReproductionNumber/Saving_Data/annepicase.png")


##