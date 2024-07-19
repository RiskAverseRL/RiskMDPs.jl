using MDPs
import Base
using RiskMDPs
using RiskMeasures
using Distributions
using Accessors
using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using CSV: File
#include("make_domains.jl")



"""
load a transient mdp from a csv file, 1-based index
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
    horizon::Integer = 300
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    # # distribution over episodes
    # edist::Vector{Float64} = ones(episodes) / episodes 
    
    # for the uniform initial state distribution, call each non-sink state equal times
    erm_ave = 0.0 
    states_number = state_count(model)
    for inistate in 1: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)
        rets = rweights' * H.rewards |> vec
        ret_erm = ERM(rets, ones(length(rets)) / length(rets), β)
        erm_ave += ret_erm * 1.0/(states_number -1)
    end
    println("Simulated ERM return: ", erm_ave)
end


###

function main()

   # π ::Vector{Int} =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    π ::Vector{Int} =[1, 1] # For the single state example
    β = 0.8
    #filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    #"data", "ruin.csv_tra.csv")
    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "single_tra.csv")
    model = load_mdp(File(filepath))
    evaluate_sim(model, π, β)

end

main()
