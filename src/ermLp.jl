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
      #print(B)
      B
  end

"""
Linear program to compute erm, exponential value function w, deterministic policy
"""
# function erm_linear_program(model::TabMDP,optimizer,β)
function erm_linear_program(model::TabMDP,B::Array,β::Real)

     lpm = Model(GLPK.Optimizer)
     set_silent(lpm)
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

    #print("\n check if the model has a dual solution\n")
    #print(has_duals(lpm))

    # Exponential value functions
    w = value.(w) 

    #Regular value functions 
    v = -1.0/β * broadcast(log,-value.(w) )
    
    # Initialize a policy and generate an optimal policy
    π = zeros(Int , state_number)
    for s in 1: state_number
        vmax = -Inf
        optimal_action = 2
        action_number = action_count(model,s)
        for a in 1: action_number
            snext = transition(model,s,a)
            temp = 0 
            for (sn, p, r) in snext
                    temp += p * v[sn]
            end
            if temp > vmax
                vmax = temp
                optimal_action = a
            end
        end
        π[s] = optimal_action
    end
    #check if there is an infeasible solution
    #println("termination status : ", termination_status(lpm))
    status = termination_status(lpm)
    # print(π)
   (w=w,v=v,π=π,status)
end


function evar_discretize2(α::Real, δ::Real, ΔR::Number)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 8*δ / ΔR^2
    βK = -log(1-α) / δ

    βs = Vector{Float64}([])
    β = β1
    while β < βK
        append!(βs, β)
        β *= log(1-α) / (β*δ + log(1-α))
    end
    
    βs
end

# Compute ERM value
function compute_erm(value_function :: Vector, initial_state_pro :: Vector)
    return sum(value_function.*initial_state_pro)
end

function main()

β = 0.05 # risk level of ERM
α = 0.8 # risk level of EVaR
δ = 0.1
ΔR =1 # how to set ΔR ?? max r - min r: r is the immediate reward

"""
Input: a csv file of a transient MDP, 1-based index
Output:  the model passed in ERM function
"""
filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")

                    
model = load_mdp(File(filepath))
βs =  evar_discretize2(α, δ, ΔR)

state_number = state_count(model)
# where is the csv file for the initial state distribution???
intial_state_pro = Vector{Float64}()
for s in 1:state_number-1
    push!(intial_state_pro,1.0/(state_number-1)) # start with a non-sink state
end
push!(intial_state_pro,0) # add the sink state with the initial probability 0

max_h =-Inf
optimal_policy = []
optimal_beta = 0.0

for β in βs
 B = compute_B(model,β)
 w,v,π,status = erm_linear_program(model,B,β)
 temp = compute_erm(v,intial_state_pro) + log(α)/β
 if temp  > max_h
    max_h = temp 
    optimal_policy = π
    optimal_beta = β
 end
end

print("\n max Evar value is  ", max_h  )
print("\n the optimal policy is  ", optimal_policy)
print("\n the optimal beta value is  ", optimal_beta)
opt_erm = max_h - log(α)/β
print("\n the optimal erm value is  ",opt_erm)

end 

main()







