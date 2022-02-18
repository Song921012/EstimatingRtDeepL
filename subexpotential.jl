using DifferentialEquations
using LinearAlgebra, DiffEqSensitivity, Optim
using Flux: flatten, params
using DiffEqFlux, Flux
using Plots
using Flux: train!
using NNlib
using LaTeXStrings

# Generate Data from subexpontial growth model

function model1(du, u, p, t)
    r, α = p
    du .= r .* u .^ α
end
u_0 = [1.0]
p_data = [0.2, 0.5]
tspan_data = (0.0, 30)
prob_data = ODEProblem(model1, u_0, tspan_data, p_data)
data_solve = solve(prob_data, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1)
data_withoutnois = Array(data_solve)
data = data_withoutnois #+ Float32(2e-1)*randn(eltype(data_withoutnois), size(data_withoutnois))
tspan_predict = (0.0, 40)
prob_predict = ODEProblem(model1, u_0, tspan_predict, p_data)
test_data = solve(prob_predict, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 1)
plot(data_solve)


ann_node = FastChain(FastDense(1, 10, tanh), FastDense(10, 1))
p = Float64.(initial_params(ann_node))
function model1_nn(du, u, p, t)
    du[1] = 0.1 * ann_node(t, p)[1] * u[1] - 0.1 * u[1]
end
prob_nn = ODEProblem(model1_nn, u_0, tspan_data, p)
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
prob_nn2 = ODEProblem(model1_nn, u_0, tspan_predict, res2_node.minimizer)
s_nn = solve(prob_nn2, Tsit5(), saveat = 1)

# I(t)
scatter(data_solve.t, data[1, :], label = "Training Data")
plot!(test_data, label = "Real Data")
plot!(s_nn, label = "Neural Networks")
xlabel!("t(day)")
ylabel!("I(t)")
title!("Subexpotential Model(I(t))")
savefig("Figures/subexpontialIt.png")
# R(t)

plot((2 ./ sqrt.(test_data) .+ 1)', label = L"R_t = 2/\sqrt{u}+1")
plot!((2 ./ sqrt.(s_nn) .+ 1)', label = L"R_t = NN(t)")
xlabel!("t(day)")
ylabel!("Effective Reproduction Number")
title!("Subexpotential Model(Rt)")
savefig("Figures/subexpontialRt.png")