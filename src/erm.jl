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
# Needs a terminal state that is a sink and has a reward 0
# Corresponds to an indefinite horizon
# ---------------------------------------------------------------

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
