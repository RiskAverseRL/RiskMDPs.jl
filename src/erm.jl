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

Save the transition matrix `B_π` for the MDP `model` and policy `π`. The matrix is used
to compute the exponential value function. The value `β` represents the risk and should be positive.  

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

# ---------------------------------------------------------------
# ERM policy iteration 
# ---------------------------------------------------------------


"""
        evf2vf!(v, β::Real, w)

Translates an exponential value function to the regular value function for a given
value of risk `β`.

The two functions are related as as
```math
\\begin{aligned}
w_s &= -\\exp(-β v_s) \\\\
v_s &= - β^{-1} \\cdot \\log(-w_s)
\\end{aligned}
```
"""
function evf2vf!(v::Vector{<:Real}, β::Real, w::AbstractVector{<:Real})
    β > zero(β) || error("Risk β must be positive.")
    v .= - one(β) / β * log.(- w)
end

"""
See `evf2vf!`
"""
function evf2vf(β, w)
    v = zeros(length(w))
    evf2vf!(v)
    v
end

"""
    policy_iteration(model, ob; [iterations=1000])

Implements policy iteration for MDP `model` with an ERM objective and discount factor `1`. The algorithm runs until the policy stops changing or the number of iterations is reached.

**Important**: Assumes that the last last is the terminal state (a self-loop)

Does not support duplicate entries in transition probabilities.
"""
function policy_iteration(model::TabMDP, obj::InfiniteERM, π₀::AbstractVector{<:Integer};
                          iterations::Int = 1000)

    # the exponential value function satisfies that
    # w_pi = B w_pi - b
    # where b is the transition to the terminal state
    # and B are the transitions between non-terminal states
    
    S = state_count(model)
    # preallocate
    w_π = fill(0., S)
    v_π = fill(0., S)
    B_π = zeros(S, S)
    IB_π = zeros(S-1, S-1) # restricted B
    b_π = zeros(S-1)

    policy = fill(-1,S)  # 2 policies to check for change
    policy .= π₀
    policyold = fill(-1,S)
    
    itercount = iterations
    for it ∈ 1:iterations      
        mrp_exp!(IB_π, model, policy)

        IB_π .= view(B_π, 1:S-1, 1:S-1)
        b_π .= view(B_π, 1:S-1, S)
        
        # Solve: v_π .= (I - IB_π) \ b_π
        _add_identity!(IB_π)
        ldiv!(w_π, lu!(IB_π), b_π)

        # convert to compute the value function
        evf2vf!(v_π, obj.β, w_π)
        
        policyold .= policy
        greedy!(policy, model, InfiniteERM, v_π)
        # check if there was a change
        if all(i->policy[i] == policyold[i], 1:S)
            itercount = it
            break
        end
    end
    (policy = policy, value = v_π, iterations = itercount)
end
