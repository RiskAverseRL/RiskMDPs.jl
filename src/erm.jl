import MDPs: qvalue
    
# ---------------------------------------------------------------
# ERM with monetary discounting: finite horizon
# ---------------------------------------------------------------

"""
Represents an ERM objective with a discount factor. It computes a Markov
policy.
"""
struct DiscountedERM <: MarkovDet
    γ::Float64  # discount factor
    β::Float64  # risk level
    T::Int      # horizon

    function DiscountedERM(γ::Number, β::Number, T::Integer)
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        β ≥ zero(β) || error("Risk β must be non-negative")
        T ≥ one(T) || error("Horizon must be at least one")
        new(γ, β, T)
    end
end

"""
    qvalue(model, obj, s, a, v)

Compute qvalue of the time-adjusted ERM risk measure. In
this model the risk level decreses with the time step
"""
function qvalue(model::MDP{S,A}, obj::DiscountedERM, t::Integer, s::S, a::A, v) where {S,A} 
    @assert t ≥ 1
    val = 0.0
    spr = getnext(model, s, a)
    # TODO: This still allocates, though less
    X = valuefunction.((model,), spr.states, (v,) )
    X *=  obj.γ
    X += spr.rewards 
    # note that the risk level decreses with the time step
    ERM(X, spr.probabilities, obj.β * (obj.γ^(t-1))) :: Float64
end

horizon(o::DiscountedERM) = o.T

# ---------------------------------------------------------------
# ERM with the total reward criterion.
# ---------------------------------------------------------------
# Needs a terminal state that is a sink and has a reward 0
# Corresponds to an indefinite horizon

"""
Represents an ERM objective with a total reward objective over
and infinite horizon. This formulation is roughly equivalent 
to using a discount factor of 1.0

This objective should be only used with the infinite horizon
version of value iteration.
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


# ---------------------------------------------------------------
# ERM policy iteration support
# ---------------------------------------------------------------

"""
    mrp!(B_π, model, π, β)

Save the transition matrix `B_π` for the 
MDP `model` and policy `π`. The matrix is used to compute the exponential
value function. The value `β` represents the risk and should be positive.  

The terminal vector must be extracted from the transition matrix separately.

Does not support duplicate entries in transition probabilities.
"""
function mrp_exp!(B_π::AbstractMatrix{<:Real}, model::TabMDP, π::AbstractVector{<:Integer}, β::Real)
    S = state_count(model)
    fill!(B_π, 0.)
    for s ∈ 1:S
        @assert !isterminal(model, s)
        for (sn, p, r) ∈ transition(model, s, π[s])
            B_π[s,sn] ≈ 0. || error("duplicated transition  ($s->$sn,...,$s->$sn)")
            B_π[s,sn] += p * exp(-β * r)
        end
    end
end

"""
    mrp_exp(model, π)

Compute the transition matrix `B_π` and the terminal vector `b_π` for the 
MDP `model` and policy `π` and risk `β`. See mrp! for more details. 
"""
function mrp_exp(model::TabMDP, π::AbstractVector{<:Integer}, β::Real)
    S = state_count(model)
    B_π = Matrix{Float64}(undef,S,S)
    mrp_exp!(B_π, model, π, β)
    B_π    
end
