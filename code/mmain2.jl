using MDPs
using CSV

include(joinpath(@__DIR__, "VarDP2.jl"))
using .VarDP2

# Load the MDP from CSV
mdp_path = joinpath(@__DIR__, "..", "data", "machine.csv")
println("Loading MDP from: $mdp_path")

mdp = VarDP2.load_intmdp(mdp_path)
S   = state_count(mdp)
println("Number of states: ", S)

T  = 10
α  = 0.1

# τ-grid based on max abs reward and horizon T
rmax = VarDP2.max_abs_reward(mdp)

# functions for grid min and max at time t
grid_min_t(t) = -t * rmax
grid_max_t(t) =  t * rmax

println("Max abs reward rmax = $rmax")
println("Tau-grid at t=T range = [$(grid_min_t(T)), $(grid_max_t(T))]")

# step size
Δτ = 5.0

X = [vcat(-Inf, collect(grid_min_t(t):Δτ:grid_max_t(t)), Inf) for t in 0:T]
println("Using horizon T = $T, α = $α")

# Run VarDP value iteration
println("\nRunning VarDP with side = :plus (h^+)…")
V_plus, π_plus = VarDP2.vi(mdp, X, T; side=:plus)
println("  V_plus size: ", size(V_plus))
println("  π_plus size: ", size(π_plus))

println("\nRunning VaR-DP with side = :minus (h^-)…")
V_minus, π_minus = VarDP2.vi(mdp, X, T; side=:minus)
println("  V_minus size: ", size(V_minus))
println("  π_minus size: ", size(π_minus))

# Compute VaR for all states
τ_plus_all  = Vector{Union{Nothing,Float64}}(undef, S)
τ_minus_all = Vector{Union{Nothing,Float64}}(undef, S)

for s in 1:S
    τ_plus_all[s]  = VarDP2.var(V_plus,  X, T, s, α)
    τ_minus_all[s] = VarDP2.var(V_minus, X, T, s, α)
end

# Print VaR_T(s) for all s
println("\n===============================================")
println("VaR thresholds at time T = $T for ALL states")
println("with confidence level 1 - α = $(1 - α):")
println("===============================================")

for s in 1:S
    τp = τ_plus_all[s]
    τm = τ_minus_all[s]
    println("State $s:")
    println("  Using h^+ (side = :plus):   τ_T^+(s) = ", isnothing(τp) ? "nothing" : string(τp))
    println("  Using h^- (side = :minus):  τ_T^-(s) = ", isnothing(τm) ? "nothing" : string(τm))
end

println("===============================================\n")

#

using Plots
s0 = 1
xs = X[T+1]
ys_plus  = vec(V_plus[T+1][s0, :])
ys_minus = vec(V_minus[T+1][s0, :])
plot(xs, ys_plus,  label="V_plus")
plot!(xs, ys_minus, label="V_minus")

