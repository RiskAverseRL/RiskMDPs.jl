
# ---------------------------------------------------------------
# EVaR with monetary discounting
# ---------------------------------------------------------------

"""
Represents a discounted EVaR objective with an explicit horizon
and explicit risk levels.
"""
struct DiscountedEVaR <: Markov
    γ::Float64

    α::Float64
    T::Int

    function DiscountedEVaR(γ::Number, α::Number, T::Integer)
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        one(α) ≥ α ≥ zero(0) || error("Risk α must be in [0,1]")
        T ≥ one(T) || error("Horizon must be at least one")
        new(γ, α, T)
    end
end


horizon(o::DiscountedEVaR) = o.T

"""
    solve_evar_points(model, objective, initstate, d_β)

Solve the EVaR `objective` for `model` and return an appropriate policy. The
initial state is the (one-based) `initstate` The discretized risk levels `d_β`
correspond to the risk level of the ERM used in the definition of EVaR.
"""
function solve_evar_points(model::TabMDP, objective::DiscountedEVaR, initstate::Int,
               d_β::AbstractVector{<:Real})
     
    zero(eltype(d_β)) ≤ minimum(d_β) || error("d_β values must be non-negative.")
    1 ≤ initstate ≤ state_count(model) ||
        error("Initstate must be between 1 and the number of states.")

    # the penalty used in the definition of EVaR
    penalty = log(1-objective.α)

    # compute objective values for the supplied points
    obj_values = similar(d_β)

    Threads.@threads for i = eachindex(d_β)
        β = d_β[i]
        erm_value = Risk.DiscountedERM(objective.γ, β, objective.T)
        # TODO: a value iteration that does not allocate would be useful here
        vp = value_iteration(model, erm_value)
        # take the value in the time step 1 in the initial state
        obj_values[i] = vp.value[1][initstate] + (1. / β) * penalty 
    end
    β_opt = d_β[argmax(obj_values)]
    #println(d_β)
    #println(obj_values)

    # recompute the value function and policy for the optimal β
    fvp = value_iteration(model, Risk.DiscountedERM(objective.γ, β_opt, objective.T))
    (value = fvp.value, policy = fvp.policy, β_opt = β_opt)
end


"""
    evar_discretize1(α, δ, ΔR)

Computes an optimal worst-case grid of β values in the definition of EVaR for
the risk level `α` (higher is more risk averse) with the error guaranteed to
be less than `δ`. The range of returns (maximum - minumum possible) must
be bounded by `ΔR`

The function uses the multiplicative step approach from the proofs rather than
the better discretization given in the paper.
"""
function evar_discretize1(α::Real, δ::Real, ΔR::Number)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 8*δ / ΔR^2
    βK =  -log(1-α) / δ
    # This is the multiplicative factor
    c = ΔR^2*log(1-α) / (8*δ^2 + ΔR^2*log(1-α))

    βs = Vector{Float64}([])
    β = β1
    while β < βK
        append!(βs, β)
        β *= c
    end
    βs
end

"""
    evar_discretize2(α, δ, ΔR)

Computes an optimal grid of β values in the definition of EVaR for
the risk level `α` (higher is more risk averse) with the error guaranteed to
be less than `δ`. The range of returns (maximum - minumum possible) must
be bounded by `ΔR`
"""
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

# ----------------------------------------------------------------
# EVaR branch and bound approach
# ----------------------------------------------------------------

using DataStructures: PriorityQueue, dequeue!
"""
Keeps track of the upper and lower bounds on each interval as used the
branch and bound method
"""
struct Interval
    index_l :: Int                  # index of the right β
    index_r :: Int                  # index of the left β
    h_upper_bound :: Float64        # upper bound on h on the interval
end
                       

"""
    solve_evar_bnb(model, objective, initstate, d_β)

Solve the EVaR `objective` for `model` and return an appropriate policy. The
initial state is the (one-based) `initstate` The discretized risk levels `d_β`
correspond to the risk level of the ERM used in the definition of EVaR.

This is a future adaptive function that will chose the points to evaluate in a way
that is more clever than the present approach.

the function v_terminal is used to compute the infinite-horizon objective by
subtituting an infinite-horizon value function at the last step of the problem. 

"""
function solve_evar_bnb(model::TabMDP, objective::DiscountedEVaR, initstate::Int,
               d_β::AbstractVector{<:Real}; v_terminal = nothing)

    issorted(d_β) || error("d_β values must be sorted increasingly") # TODO: check also increasing
    zero(eltype(d_β)) ≤ d_β[1] || error("d_β values must be non-negative.")
    zero(eltype(d_β)) ≤ d_β[end] || error("One d_β must be positive.")
    
    1 ≤ initstate ≤ state_count(model) ||
        error("Initstate must be between 1 and the number of states.")
    
    # the penalty used in the definition of EVaR
    penalty = log(1-objective.α)
    erm_values = similar(d_β)

    # global model for value function and policy to avoid excessive
    # memory allocation
    vp_temp = make_value(model, Risk.DiscountedERM(objective.γ, 0.0, objective.T))
    
    # evaluate the ERM value for an index
    function eval_erm(index::Int)
        local erm_value = Risk.DiscountedERM(objective.γ, d_β[index], objective.T)
        value_iteration!(vp_temp.value, vp_temp.policy, model, erm_value; v_terminal = v_terminal)
        return vp_temp.value[1][initstate] 
    end

    function make_interval(index_l::Int, index_r::Int)
        @assert index_l < index_r
        upperbound =  erm_values[index_l] + 1. / d_β[index_r] * penalty
        return Interval(index_l, index_r, upperbound)
    end

    # Holds the different interval values
    intervals = PriorityQueue{Interval,Float64}(Base.Order.Reverse) # maximum element first

    # compute extreme points of the initial interval
    erm_values[1] = eval_erm(1)
    erm_values[length(d_β)] = eval_erm(length(d_β))

    # compute lower bounds
    lower_bound = erm_values[end] + (1. / d_β[end]) * penalty
    best_index = length(d_β)

    let obj = erm_values[1] + (d_β[1] > 0. ? (1. / d_β[1]) * penalty : 0.)
        if(obj > lower_bound)
            lower_bound = obj
            best_index = 1
        end
    end
        
    init = make_interval(1, length(d_β))
    push!(intervals, (init => init.h_upper_bound))

    eval_count = 2
    # do not parallelize, there would be a race due to vp_temp
    while !isempty(intervals)
        eval_count += 1
        #println(lower_bound, intervals)
        i = dequeue!(intervals)

        if i.index_r ≤ i.index_l + 1
            continue
        end

        # best element is worse, no improvement possible
        if i.h_upper_bound ≤ lower_bound
            break
        end

        nextindex :: Int = (i.index_l + i.index_r) ÷ 2 # TODO: could overflow ...

        erm_values[nextindex] = eval_erm(nextindex)
        let obj = erm_values[nextindex] + 1. / d_β[nextindex] * penalty
            if(obj > lower_bound)
                lower_bound = obj
                best_index = nextindex
            end
        end

        int_l = make_interval(i.index_l, nextindex)
        if(int_l.h_upper_bound > lower_bound)
            push!(intervals, (int_l => int_l.h_upper_bound))
        end
        int_r = make_interval(nextindex, i.index_r)
        if(int_r.h_upper_bound > lower_bound)
            push!(intervals, (int_r => int_r.h_upper_bound))
        end
    end

    # recompute the value function and policy for the optimal β
    β_opt = d_β[best_index]
    value_iteration!(vp_temp.value, vp_temp.policy,
                     model, Risk.DiscountedERM(objective.γ, β_opt, objective.T);
                     v_terminal = v_terminal)
    (value = vp_temp.value, policy = vp_temp.policy, β_opt = β_opt, eval_count = eval_count)
end

# ------------------------------------------------------
# Other EVaR methods used for comparison purposes mainly
# -------------------------------------------------------

"""
    solve_evar_points_nested(model, objective)

Solve the EVaR `objective` for `model` and return an appropriate policy. The
initial state is the (one-based) `initstate` The discretized risk levels `d_β`
correspond to the risk level of the ERM used in the definition of EVaR.

WARNING: This method is just to study ablation. There is probably no good reason
to actually use it. 
"""
function solve_evar_points_nested(model::TabMDP, objective::DiscountedEVaR, initstate::Int,
               d_β::AbstractVector{<:Real})
    println("WARNING: You should probably be using solve_evar_points instead")
    
    zero(eltype(d_β)) ≤ minimum(d_β) || error("d_β values must be non-negative.")
    1 ≤ initstate ≤ state_count(model) ||
        error("Initstate must be between 1 and the number of states.")

    # the penalty used in the definition of EVaR
    penalty = log(1-objective.α)

    # compute objective values for the supplied points
    obj_values = similar(d_β)

    Threads.@threads for i = eachindex(obj_values, d_β)
        β = d_β[i]
        erm_value = Risk.NestedFiniteH(objective.γ, (X,p) -> erm(X,p, β), objective.T)
        # TODO: a value iteration that does not allocate would be useful here
        vp = value_iteration(model, erm_value)
        # take the value in the time set 1 in the initial state
        obj_values[i] = vp.value[1][initstate] + (1. / β) * penalty 
    end
    β_opt = d_β[argmax(obj_values)]

    # recompute the value function and policy
    value_iteration(model,
       Risk.NestedFiniteH(objective.γ, (X,p) -> erm(X,p, β_opt),
                        objective.T))
end
