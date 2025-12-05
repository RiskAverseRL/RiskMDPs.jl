using MDPs
using CSV

include(joinpath(@__DIR__, "VarDP.jl"))
using .VarDP

# Load the MDP from CSV
mdp_path = joinpath(@__DIR__, "..", "data", "inventory2.csv")
println("Loading MDP from: $mdp_path")

mdp = VarDP.load_intmdp(mdp_path)
S   = state_count(mdp)
println("Number of states: ", S)

T  = 10         
α  = 0.1
s0 = 20

# τ-grid based on max abs reward and horizon T
rmax = VarDP.max_abs_reward(mdp)
grid_min = -T * rmax
grid_max =  T * rmax

println("Max abs reward rmax = $rmax")
println("Tau-grid range = [$grid_min, $grid_max]")

# step size 
Δτ = 50.0

X = vcat(-Inf, collect(grid_min:Δτ:grid_max), Inf)
K = length(X)
println("Grid size K = $K")



println("Using horizon T = $T, α = $α, initial state s0 = $s0")

# Run VarDP value iteration 
println("\nRunning VarDP with side = :plus (h^+)…")
V_plus, π_plus = VarDP.vi(mdp, X, T; side=:plus)

println("  V_plus size: ", size(V_plus))   # (T+1, S, K)
println("  π_plus size: ", size(π_plus))   # (T,   S, K)

τ_plus = VarDP.var(V_plus, X, T, s0, α)


println("\nRunning VaR-DP with side = :minus (h^-)…")
V_minus, π_minus = VarDP.vi(mdp, X, T; side=:minus)

println("  V_minus size: ", size(V_minus)) # (T+1, S, K)
println("  π_minus size: ", size(π_minus)) # (T,   S, K)

τ_minus = VarDP.var(V_minus, X, T, s0, α)

# Print VaR_T(s0) 
println("\n===============================================")
println("VaR thresholds at time T = $T from initial state s0 = $s0")
println("with confidence level 1 - α = $(1 - α):")
println("  Using h^+ (side = :plus):   τ_T^+(s0) = ",
        isnothing(τ_plus)  ? "nothing" : string(τ_plus))
println("  Using h^- (side = :minus):  τ_T^-(s0) = ",
        isnothing(τ_minus) ? "nothing" : string(τ_minus))
println("===============================================\n")

###
if false
###
    using Plots
    plot(X, V_plus[T, s0, :])
    plot!(X, V_minus[T, s0, :])
###    
end
