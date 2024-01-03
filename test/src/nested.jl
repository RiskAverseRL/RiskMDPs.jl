
#using MDPs
#using RiskMDPs
using RiskMeasures
#using Distributions
#include("make_domains.jl")
#prob = TestDomains["onestatepm"]

# evaluates the policy by simulation
function evaluate_sim(prob::Problem, π)

    # evaluation helper variables
    episodes = 10000
    rweights::Vector{Float64} = prob.γ .^ (0:prob.horizon-1)     # reward weights
    edist::Vector{Float64} = ones(episodes) / episodes # distribution over episodes
    
    H = simulate(prob.model, π, prob.initstate, prob.horizon, episodes)
    returns = rweights' * H.rewards |> vec
    return sum(returns) / length(returns)
end

@testset "Nested Simple" begin
    for (probname, prob) ∈ TestDomains
        #println(probname)
        # Risk-neutral finite 
        vp = value_iteration(prob.model, FiniteH(prob.γ, prob.horizon))
        #println("Expected return ", vp.value[1][prob.initstate])
        v = vp.value
        π = vp.policy
        ret_mean = evaluate_sim(prob, π)
        #println(ret_mean)

        # Nested CVaR infinite
        # -- just use finite horizon for now
        #obj = NestedInfiniteH(γ, (X,p) -> cvar(X,p,α).cvar)
        #time = @elapsed v = value_iteration(model, obj)
        #π = greedy(model, obj, v.value)
        #report_disc!(results, "Nested CVaR", π, v, time)

        # Nested CVaR
        obj = NestedFiniteH(prob.γ, prob.horizon,
                            (X,p) -> CVaR_e(X,p,prob.α).value)
        vp = value_iteration(prob.model, obj)
        v = vp.value
        π = vp.policy

        ret_cvar = evaluate_sim(prob, π)
        #println(ret_cvar)
        #println(vp.value[1][prob.initstate])

        # ret_mean should be computed using the optimal policy
        @test ret_cvar ≤ ret_mean + 0.1 * abs(ret_mean) + 0.1
        
        # this is not very meaningful, but it is something
        @test ret_cvar ≥ vp.value[1][prob.initstate] - 1e-2 # deterministic
    end
end
