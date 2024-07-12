import MDPs: qvalue
import Base
using DataFrames: DataFrame
using DataFramesMeta
using Revise
using MDPs
using LinearAlgebra
using GLPK
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
# assume a determinstic policy, that is, d_a(s) is always 1.
function compute_B(model::TabMDP,β::Real)
    
    states_size = state_count(model)
    actions_size = maximum([action_count(model,s) for s in 1:states_size])

    # Initialize B[s,a,sn]
    B = zeros(Float64 , states_size, actions_size,states_size)
     
    #calculate B[s,s',a], including a sink state, assume a deterministic policy
       for s in 1: states_size
          action_number = action_count(model,s)
          for a in 1: action_number
              snext = transition(model,s,a)
              for (sn, p, r) in snext
                  B[s,a,sn] = p  * exp(-β * r) 
              end
          end
      end 
      B
  end

"""
Linear program to compute erm, exponential value function w
"""
# function erm_linear_program(model::TabMDP,optimizer,β)
function erm_linear_program(model::TabMDP,B::Array,β::Real)
     lpm = Model(GLPK.Optimizer)
     #set_silent(lpm)
     state_number = state_count(model)
     
     @variable(lpm,w[1: state_number] )
     @objective(lpm,Min,sum(w[1: state_number]))
    
     # assume that the last state is the sink state, constraint for the sink state
     @constraint(lpm, constraint1,w[state_number] == -1)

    # construct constraints for non-sink states, the sink state = state_number
    for s in 1: state_number-1
        action_number = action_count(model,s)
        for a in 1: action_number
            snext = transition(model,s,a)
            # compute B_{s,̇̇}^a * \bm{w}
            bw = 0 
            for (sn, p, r) in snext
                if sn != state_number # state_number is the sink state
                    bw += B[s,a,sn] *w[sn]
                end
            end
            @constraint(lpm, w[s] ≥ -B[s,a,state_number] + bw )
        end
    end

    optimize!(lpm)
    # output exponential value functions
    print("\n Exponential value functions\n")
    print(value.(w))  
    #output regular value functions 
    print("\n Regular value functions\n")
    print(-1.0/β * broadcast(log,-value.(w) ) )     
end


function main()

β = 0.1

"""
Input: a csv file of a transient MDP, 1-based index
Output:  the model passed in ERM function
"""
filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")

                    
model = load_mdp(File(filepath))
B = compute_B(model,β)
erm_linear_program(model,B,β)




end 

main()







