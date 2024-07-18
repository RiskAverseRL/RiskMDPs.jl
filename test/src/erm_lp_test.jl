using Revise
using MDPs
import Base
using RiskMDPs
using RiskMeasures
using Distributions
using Accessors
using DataFrames: DataFrame
using DataFramesMeta
using Revise
using LinearAlgebra
using CSV: File
using Infiltrator
#include("make_domains.jl")



"""
load mdp from a csv file, 1-based index
"""
function load_mdp(input)
    mdp = DataFrame(input)
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)
    
    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{IntState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{IntAction}(undef, maximum(sd.idaction))
       
        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            try 
            actions[idaction] = IntAction(ad.idstateto, ad.probability, ad.reward)
            catch e
                error("Error in state $(idstate-1), action $(idaction-1): $e")
            end
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                " that were uninitialized " * string(findall(.!action_init) .- 1 ) ))

        states[idstate] = IntState(actions)
        state_init[idstate] = true
    end
    IntMDP(states)
end


####

# evaluates the policy by simulation
function evaluate_sim(model::TabMDP, π::Vector{Int}, β::Real)
    # evaluation helper variables
    episodes = 1000
    horizon = 500
    # reward weights
    rweights::Vector{Float64} = 1 .^ (0:horizon-1)     
    # distribution over episodes
    edist::Vector{Float64} = ones(episodes) / episodes 
    
    # for the uniform initial state distribution, call each non-sink state equal times
    inistate::Int64 = 10
    H = simulate(model, π, inistate, horizon, episodes)

    initial = ones(state_count(model))
    initial[end] = 0
    initial /= sum(initial)
    
    H = simulate(model, π, initial, horizon, episodes)
    #@infiltrate
    rets = rweights' * H.rewards |> vec
    ret_erm = ERM(rets, ones(length(rets)) / length(rets), β)
    println("Simulated ERM return: ", ret_erm)
end

    # max Evar value is  3.430243025414316
    #the optimal policy is  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # the optimal beta value is  0.8
    # the optimal erm value is  7.89311405169851

###

function main()

    π ::Vector{Int} =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    β = 0.8
    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")
    model = load_mdp(File(filepath))
    evaluate_sim(model, π, β)

end

main()
