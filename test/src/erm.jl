using Revise
using MDPs
using RiskMDPs
using RiskMeasures
using Distributions
include("make_domains.jl")
prob = TestDomains["onestatepm"]

# evaluates the policy by simulation
function evaluate_sim(prob::Problem, π, β)
    # evaluation helper variables
    episodes = 100000
    # reward weights
    rweights::Vector{Float64} = prob.γ .^ (0:prob.horizon-1)     
    # distribution over episodes
    edist::Vector{Float64} = ones(episodes) / episodes 
    
    H = simulate(prob.model, π, prob.initstate, prob.horizon, episodes)
    returns = rweights' * H.rewards |> vec
    return ERM(returns, ones(length(returns)) / length(returns), β)
end

#@testset "Nested Simple" begin
    for (probname, prob) ∈ TestDomains
        println(probname)

        # Discounted ERM
        obj = DiscountedERM(prob.γ, prob.β, prob.horizon)
        vp = value_iteration(prob.model, obj)
        v = vp.value
        π = vp.policy

        ret_erm = evaluate_sim(prob, π, prob.β)
        println("Expected ERM return: ", v[1][prob.initstate])
        println("Simulated ERM return: ", ret_erm)

        # ret_mean should be computed using the optimal policy
        #@test ret_cvar ≤ ret_mean + 0.1 * abs(ret_mean) + 0.1
        
        # this is not very meaningful, but it is something
        #@test ret_cvar ≥ vp.value[1][prob.initstate] - 1e-2 # deterministic
    end
#end
