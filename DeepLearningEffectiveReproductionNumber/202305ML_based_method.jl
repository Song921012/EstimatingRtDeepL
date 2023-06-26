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
display(plot(data_daily, label="Daily Confirmed Cases", lw=2))
display(plot(data_acc, label="Accumulated Confirmed Cases", lw=2))
data_daily[1]
println(length(data_acc))
trainingdata = Float32.(data_acc)
# set up neural differential equation models


ann = Lux.Chain(Lux.Dense(1, 32, tanh), Lux.Dense(32, 1))
p_0, st = Lux.setup(rng, ann)
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

predict_neuralode(p_0)[2, :]
size(predict_neuralode(p_0)[2, :]) == size(data_acc)
# loss function and callbacks

function loss_neuralode(p)
    pred = predict_neuralode(p)[2, :]
    loss = sum(abs2, log.(trainingdata) .- log.(pred))
    return loss, pred
end

loss_neuralode(p_0)


callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata, label="Accumulated cases")
        plot!(plt, tsteps, pred, label="Predicted accumulated cases")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p_0)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
##
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.ADAM(0.05),
    callback=callback,
    maxiters=300)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)


pfinal = result_neuralode2.u

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)


# Save neural network architechtures and 
using BSON: @save
@save "./DeepLearningEffectiveReproductionNumber/Saving_Data/ann_nn_ir.bson" ann
psave = collect(pfinal)
@save "./DeepLearningEffectiveReproductionNumber/Saving_Data/ann_para_irlbfgs.bson" psave
pred = predict_neuralode(pfinal)[2, :]
plt = scatter(tsteps, trainingdata, label="Accumulated cases")
plot!(plt, tsteps, pred, label="Predicted accumulated cases")
display(plot(plt))
savefig("./DeepLearningEffectiveReproductionNumber/Saving_Data/annepicase.png")


##