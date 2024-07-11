import MDPs: qvalue
import Base
using DataFrames: DataFrame
using DataFramesMeta
using Revise
using MDPs
using LinearAlgebra
using JuMP
using CSV: File
using RiskMDPs



# ---------------------------------------------------------------
# ERM with the total reward criterion.
# ---------------------------------------------------------------
# Needs a terminal state that is a sink and has a reward 0
# Corresponds to an indefinite horizon

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
    """
    # create transitions to itself for each uninitialized state
    # to simulate a terminal state
    for s ∈ findall(.!state_init)
        states[s] = IntState([IntAction([s], [1.], [0.])])
    end
    """
    IntMDP(states)
end

"""
Input: a csv file of a transient MDP, 1-based index
Output:  the model passed in ERM function
"""
filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")
                    
model = load_mdp(File(filepath))
# print("state count  ")
# print(state_count(model))

"""
Represents an ERM objective with a total reward objective over
an infinite horizon. This formulation is roughly equivalent 
to using a discount factor of 1.0
"""
struct InfiniteERM <: StationaryDet
    β::Float64  # risk level

    function InfiniteERM(β::Number)
        β ≥ zero(β) || error("Risk β must be non-negative")
        new(β)
    end
end

"""
    qvalue(model, obj, s, a, v)

Compute qvalue of the time-adjusted ERM risk measure. Note that this
qvalue must be time-dependent.
"""

"""
function qvalue(model::MDP{S,A}, obj::InfiniteERM, s::S, a::A, v) where {S,A} 
    val = 0.0
    spr = getnext(model, s, a)
    # TODO: This still allocates, though less
    X = valuefunction.((model,), spr.states, (v,) )
    X += spr.rewards 
    ERM(X, spr.probabilities, obj.β) :: Float64
end

horizon(o::InfiniteERM) = o.T
discount(o::InfiniteERM) = 1.0

"""