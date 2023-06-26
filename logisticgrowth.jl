##
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DataFrames
using CSV
using ComponentArrays
using OptimizationOptimisers
using Flux
using Plots
using LaTeXStrings
rng = Random.default_rng()
Random.seed!(14);


# Gen
##
function model2(du, u, p, t)
    r, α = p
    du .= r .* u .* (1 .- u ./ α)
end
u_0 = [1.0]
p_data = [0.2, 30]
tspan_data = (0.0, 30)
prob_data = ODEProblem(model2, u_0, tspan_data, p_data)
data_solve = solve(prob_data, Tsit5(), abstol=1e-12, reltol=1e-12, saveat=1)
data_withoutnois = Array(data_solve)
data = data_withoutnois #+ Float32(2e-1)*randn(eltype(data_withoutnois), size(data_withoutnois))
tspan_predict = (0.0, 40)
prob_predict = ODEProblem(model2, u_0, tspan_predict, p_data)
test_data = solve(prob_predict, Tsit5(), abstol=1e-12, reltol=1e-12, saveat=1)
plot(test_data)

#ann_node = FastChain(FastDense(1, 10, tanh), FastDense(10, 1))
#p = Float64.(initial_params(ann_node))


##
ann_node = Lux.Chain(Lux.Dense(1, 10, tanh), Lux.Dense(10, 1))
p, st = Lux.setup(rng, ann_node)
function model2_nn(du, u, p, t)
    du[1] = 0.1 * ann_node([t], p, st)[1][1] * u[1] - 0.1 * u[1]
end
prob_nn = ODEProblem(model2_nn, u_0, tspan_data, ComponentArray(p))
function train(θ)
    Array(concrete_solve(prob_nn, Tsit5(), u_0, θ, saveat=1,
        abstol=1e-6, reltol=1e-6))#,sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end
#println(train(p))
function loss(θ)
    pred = train(θ)
    sum(abs2, (data .- pred)), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

const losses = []
callback(θ, l, pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end

pinit = ComponentArray(p)
println(loss(p))
callback(pinit, loss(pinit)...)


##
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.ADAM(0.05),
    callback=callback,
    maxiters=500)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)


pfinal = result_neuralode2.u

println(pfinal)
prob_nn2 = ODEProblem(model2_nn, u_0, tspan_predict, pfinal)
s_nn = solve(prob_nn2, Tsit5(), saveat=1)

# I(t)
scatter(data_solve.t, data[1, :], label="Training Data")
plot!(test_data, label="Real Data")
plot!(s_nn, label="Neural Networks")
xlabel!("t(day)")
ylabel!("I(t)")
title!("Logistic Growth Model(I(t))")
savefig("Figures/logisticIt.png")
# R(t)
f(x) = 2 * (1 - x / p_data[2]) + 1
plot((f.(test_data))', label=L"R_t = 2(1-I(t)/K)+1")
plot!((f.(s_nn))', label=L"R_t = NN(t)")
xlabel!("t(day)")
ylabel!("Effective Reproduction Number")
title!("Logistic Growth Model(Rt)")
savefig("Figures/logisticRt.png")