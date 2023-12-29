# --------------------------------------------------------------
# Nested risk: infinite horizon
# --------------------------------------------------------------

"""
Represents a discounted infinite horizon objective with an iterated
risk measure.

The function `risk` maps a vector of values and probabilities to the
risk value
"""
struct NestedInfiniteH <: Stationary
    γ::Float64
    risk::Function

    function NestedInfiniteH(γ::Number, risk::Function) 
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        new(γ, risk)
    end
end

"""
    qvalue(model, obj::NestedInfiniteH, s, a, v)

Compute qvalue of a nested (or iterated) risk-averse objective.
"""
@inline function qvalue(model::MDP{S,A}, obj::NestedInfiniteH, s::S, a::A, v) where {S,A} 
    val = 0.0
    # TODO: this allocates memory
    X = Vector{Float64}() # random variable
    P = Vector{Float64}()
    for (sn, p, r) ∈ transition(model, s, a)
        append!(P, p)
        append!(X, r + obj.γ * valuefunction(model, sn, v))
    end
    obj.risk(X, P) :: Float64
end


# ---------------------------------------------------------------
# Nested risk: finite horizon 
# ---------------------------------------------------------------

"""
Represents a nested risk objective with a discount factor. It computes a Markov policy
for a finite horizon and not a stationary policy. 
"""
struct NestedFiniteH <: Markov
    γ::Float64
    risk::Function
    T::Int

    function NestedFiniteH(γ::Number, risk::Function, T::Integer)
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        T ≥ one(T) || error("Horizon must be positive")
        # TODO: write a test to make sure that the horizon's definition is correct
        new(γ, risk, T)
    end
end

"""
    qvalue(model, t, obj, s, a, v)

Compute qvalue of the time unadjusted ERM risk measure.
"""
@inline function qvalue(model::MDP{S,A}, t::Integer, obj::NestedFiniteH,
                        s::S, a::A, v) where {S,A} 
    val = 0.0
    # TODO: this allocates memory
    X = Vector{Float64}() # random variable
    P = Vector{Float64}() # probability distribution
    for (sn, p, r) ∈ transition(model, s, a)
        append!(P, p)
        append!(X, r + obj.γ * valuefunction(model, sn, v))
    end
    obj.risk(X, P) :: Float64
end

horizon(o::NestedFiniteH) = o.T
