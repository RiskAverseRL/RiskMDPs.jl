
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
    qvalue(model, γ, s, a, v)

Compute qvalue of the time-adjusted ERM risk measure. In
this model the risk level decreses with the time step
"""
@inline function qvalue(model::MDP{S,A}, t::Integer, obj::DiscountedERM,
                        s::S, a::A, v) where {S,A} 
    @assert t ≥ 1
    val = 0.0
    spr = getnext(model, s, a)
    # TODO: This still allocates, though less
    X = valuefunction.((model,), spr.states, (v,) )
    X *=  obj.γ
    X += spr.rewards 
    # note that the risk level decreses with the time step
    erm(X, spr.probabilities, obj.β * (obj.γ^(t-1))) :: Float64
end

horizon(o::DiscountedERM) = o.T

# ---------------------------------------------------------------
# ERM with termination discounting (indefinite horizon): finite horizon
# ---------------------------------------------------------------

"""
Represents an ERM objective with a discount factor that is interpreted.
as the probability of not-terminating. It computes a Markov policy.
"""
struct IndefiniteERM <: MarkovDet
    γ::Float64  # discount factor = probability of NOT terminating
    β::Float64  # risk level
    T::Int      # horizon

    function IndefiniteERM(γ::Number, β::Number, T::Integer)
        one(γ) ≥ γ > zero(γ) || error("Discount γ must be in (0,1]")
        β ≥ zero(β) || error("Risk β must be non-negative")
        T ≥ one(T) || error("Horizon must be at least one")
        new(γ, β, T)
    end
end

"""
    qvalue(model, γ, s, a, v)

Compute qvalue of the time-adjusted ERM risk measure.
"""
@inline function qvalue(model::MDP{S,A}, t::Integer, obj::IndefiniteERM,
                        s::S, a::A, v) where {S,A} 
    @assert t ≥ 1
    val = 0.0
    spr = getnext(model, s, a)
    # TODO: This still allocates, though less
    X = valuefunction.((model,), spr.states, (v,) )
    X *=  obj.γ
    X += spr.rewards 
    erm(X, spr.probabilities, obj.β) - (1/β) * log(γ) :: Float64
end

horizon(o::IndefiniteERM) = o.T
