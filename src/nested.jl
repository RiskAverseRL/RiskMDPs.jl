import MDPs: horizon, discount, qvalue

# --------------------------------------------------------------
# Nested risk: infinite horizon
# --------------------------------------------------------------

"""
Represents a discounted infinite horizon objective with an iterated
risk measure. Considers only deterministic policies.

The function `risk` maps a vector of values and probabilities to the
risk value
"""
struct NestedInfiniteH <: StationaryDet
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
function qvalue(model::MDP{S,A}, obj::NestedInfiniteH, s::S, a::A, v) where {S,A} 
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

discount(o::NestedInfiniteH) = o.γ

# ---------------------------------------------------------------
# Nested risk: finite horizon 
# ---------------------------------------------------------------

"""
A nested risk objective with a discount factor. It computes a Markov policy
for a finite horizon and not a stationary policy. 
"""
struct NestedFiniteH <: MarkovDet
    γ::Float64
    T::Int
    risk::Function

    function NestedFiniteH(γ::Number, T::Integer, risk::Function)
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        T ≥ one(T) || error("Horizon must be positive")
        new(γ, T, risk)
    end
end

"""
    qvalue(model, t, obj, s, a, v)

Compute the qvalue of a nested risk measure.
"""
function qvalue(model::MDP{S,A}, obj::NestedFiniteH, s::S, a::A, v) where {S,A} 
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
discount(o::NestedFiniteH) = o.γ
