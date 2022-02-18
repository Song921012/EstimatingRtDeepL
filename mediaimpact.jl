using DifferentialEquations
using LinearAlgebra, DiffEqSensitivity, Optim
using Flux: flatten, params
using DiffEqFlux, Flux
using Plots
using Flux: train!
using NNlib
using LaTeXStrings

# Generate Data from subexpontial growth model

function model3(du, u, p, t)
    r, α = p
    du .= r .* u .* exp.(-r .* u) .* (1 .- u ./ α) 
end
u_0 = [1.0]
p_data = [0.2, 30]
tspan_data = (0.0, 20)
prob_data = ODEProblem(model3, u_0, tspan_data, p_data)
data_solve = solve(prob_data, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1)
data_withoutnois = Array(data_solve)
data = data_withoutnois #+ Float32(2e-1)*randn(eltype(data_withoutnois), size(data_withoutnois))
tspan_predict = (0.0, 40)
prob_predict = ODEProblem(model3, u_0, tspan_predict, p_data)
test_data = solve(prob_predict, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1)
plot(test_data)


ann_node = FastChain(FastDense(1, 10, tanh), FastDense(10, 1))
p = Float64.(initial_params(ann_node))
function model3_nn(du, u, p, t)
    du[1] = 0.1 * ann_node(t, p)[1] * u[1] - 0.1 * u[1]
end
prob_nn = ODEProblem(model3_nn, u_0, tspan_data, p)
function train(θ)
    Array(concrete_solve(prob_nn, Tsit5(), u_0, θ, saveat = 1,
        abstol = 1e-6, reltol = 1e-6))#,sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end
#println(train(p))
function loss(θ)
    pred = train(θ)
    sum(abs2, (data .- pred)), pred # + 1e-5*sum(sum.(abs, params(ann)))
end
#println(loss(p))
const losses = []
callback(θ, l, pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end
res1_node = DiffEqFlux.sciml_train(loss, p, ADAM(0.02), cb = callback, maxiters = 500)
res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm = 0.01), cb = callback, maxiters = 2000)
println(res2_node.minimizer)
prob_nn2 = ODEProblem(model3_nn, u_0, tspan_predict, res2_node.minimizer)
s_nn = solve(prob_nn2, Tsit5(), saveat = 1)

# I(t)
scatter(data_solve.t, data[1, :], label = "Training Data")
plot!(test_data, label = "Real Data")
plot!(s_nn, label = "Neural Networks")
xlabel!("t(day)")
ylabel!("I(t)")
title!("Media Impact Model(I(t))")
savefig("Figures/mediaimpactIt.png")
# R(t)
r, α = p_data
f(x) = r * exp(-r * x) * (1 - x / α)
plot((f.(test_data))', label = L"R_t = 0.2* exp(-0.2 * x) * (1 - x/K)")
plot!((f.(s_nn))', label = L"R_t = NN(t)")
xlabel!("t(day)")
ylabel!("Effective Reproduction Number")
title!("Media Impact Model(Rt)")
savefig("Figures/mediaimpactRt.png")