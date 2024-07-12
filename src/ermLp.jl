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
# The last state is the sink state

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


"""
Compute B[s,s',a],  b_s^a, B_s^a, d_a(s) 
"""
# function compute_B(model::TabMDP,obj::InfiniteERM)
function compute_B(model::TabMDP,β::Real)
    # Initialize B[s,sn]

    #Compute d_a(s)
      
       #calculate B[s,s',a]
       # s can be a sink state
       for s in 1: state_number
          action_number = action_count(model,s)
          for a in 1: action_number
              snext = transition(model,s,a)
              # Calculate b_s^a and B_s^a
              for (sn, p, r) in snext
                  # how to get d_a(s)???
                  B[s,sn,a] += p * d_a(s) *exp(β * r) 
                  # b_s^a is a special case of B[s,sn], and sn is considered as a sink state
              end
          end
      end 
  
  end



function main()

β = 0.01

"""
Input: a csv file of a transient MDP, 1-based index
Output:  the model passed in ERM function
"""
filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")
                    
model = load_mdp(File(filepath))
# print("state count  ")
# print(state_count(model))

compute_B(model,β)




end 

main()

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





"""
Use linear program to compute erm 

"""

"""
function erm_linear_program(model::TabMDP,objective::InfiniteH,optimizer,β)
     lpm = Model(optimizer)
     set_silent(lpm)
     state_number = state_count(model)
     # exponential value function w
     @variable(lpm,w[1:n])
     @objetive(lpm,Min,sum(w[1:n]))
    
          
         
    # construct constraints
    for s in 1: state_number
        action_number = action_count(model,s)
        for a in 1: action_number
            snext = transition(model,s,a)
            # Calculate b_s^a and B_s^a
            for (sn, p, r) in snext        
                @constraint(lpm, ineqconstraint,w[s] ≥ -B[s,e,a] + sum(B[s,sn,a] *w[sn]  for (sn, p, r) in snext ) )
                # assume that the last state is the sink state
                @constraint(m, eqconstraint1,w[state_number] == -1)
            end
        end
    end

    optimize!(lpm)
    return value.(w)         
end
"""