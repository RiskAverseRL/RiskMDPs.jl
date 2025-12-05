module VarDP

using MDPs
using CSV

# Approximation functions: hp and hm on a grid x

struct Mesh
    xs::Vector{Float64}
    function Mesh(xs::AbstractVector{<:Real})
        v = collect(float.(xs))
        @assert issorted(v)
        @assert first(v) == -Inf
        @assert last(v)  ==  Inf
        new(v)
    end
end

# Indices for h^-(x) and h^+(x) on the mesh
@inline function hm(mesh::Mesh, x::Float64)
    i = searchsortedlast(mesh.xs, x)     # largest index i with xs[i] ≤ x
    i < 1 ? 1 : i
end

@inline function hp(mesh::Mesh, x::Float64)
    i = searchsortedfirst(mesh.xs, x)    # smallest index i with xs[i] ≥ x
    i > length(mesh.xs) ? length(mesh.xs) : i
end

"""
  hmin_idx(x)  = argmax{i: X[i] ≤ x}
  hplus_idx(x) = argmin{i: X[i] ≥ x}
"""
function h_funcs(X::Vector{Float64})
    mesh = Mesh(X)
    hm_idx(x::Float64) = hm(mesh, x)
    hp_idx(x::Float64) = hp(mesh, x)
    return hm_idx, hp_idx
end

"""
    load_intmdp(path; idoutcome=nothing, docompress=false)

Load an IntMDP from a CSV file with columns:
  idstatefrom, idaction, idstateto, probability, reward

Since my csv already 1-based, so I call `load_mdp` with `zerobased=false`.
"""
function load_intmdp(path::AbstractString; idoutcome=nothing, docompress=false)
    return MDPs.load_mdp(
        CSV.File(path);
        idoutcome = idoutcome,
        zerobased = false,
        docompress = docompress,
    )
end


"""
Max absolute reward in an IntMDP.
"""
function max_abs_reward(mdp::IntMDP)
    max_r = 0.0
    S = state_count(mdp)

    for s in 1:S
        for a in actions(mdp, s)
            nxt = getnext(mdp, s, a)
            rewards = nxt.rewards
            for r in rewards
                max_r = max(max_r, abs(r))
            end
        end
    end
    return max_r
end


"""
    vi(mdp, X, T; γ=0.9, side=:plus)

Value iteration on IntMDP with VarDP approximation.

Arguments:
    mdp :: IntMDP -- the MDP model loaded via `load_intmdp`
    X   :: Vector{Float64} -- grid for VarDP approximation ( -∞ to +∞ )
    T   :: Int -- time horizon
    γ   :: Float64 -- discount factor
    side :: Symbol -- :plus for h^+, :minus for h^-

Returns:
    V :: Array{Float64, 3} with size (T+1, S, K)
        where V[t+1, s, k] = v_t(s, X[k])

    π :: Array{Int, 3} with size (T, S, K)
        where π[t, s, k] = greedy action at time t in state s with threshold X[k]
"""
function vi(mdp::IntMDP,
            X::Vector{Float64},
            T::Int;
            side::Symbol = :plus)

    S = state_count(mdp)
    K = length(X)

    hm_idx, hp_idx = h_funcs(X)

    # V[t+1, s, k] = v_t(s, X[k])
    V = zeros(Float64, T+1, S, K)

    # greedy policy: π[t, s, k] for t = 0 to T-1
    π = zeros(Int, T, S, K)

    # t = 0 : V_0 (s,x) = 1{x ≥ 0}
    for s ∈ 1:S, k ∈ 1:K
        V[1, s, k] = X[k] ≥ 0.0 ? 1.0 : 0.0
    end

    # Backward recursion
    for t ∈ 1:T
        for s ∈ 1:S, k ∈ 1:K
            x = X[k]

            max_val = -Inf
            best_a  = first(actions(mdp, s))

            for a ∈ actions(mdp, s)
                nxt = getnext(mdp, s, a)
                states_ns = nxt.states
                probs     = nxt.probabilities
                rewards   = nxt.rewards

                exp_val = 0.0
                @inbounds for i ∈ eachindex(states_ns, probs, rewards)
                    ns  = states_ns[i]
                    p   = probs[i]
                    p == 0.0 && continue

                    r_sa = rewards[i]
                    nx   = x + r_sa
                    j    = (side == :plus) ? hp_idx(nx) : hm_idx(nx)

                    exp_val += p * V[t, ns, j]
                end


                if exp_val > max_val
                    max_val = exp_val
                    best_a  = a
                end
            end
            V[t+1, s, k] = max_val
            π[t, s, k]   = best_a
        end
    end
    return V, π
end

"""
    var(v, X, t, s, α)

τ_t(s) = max{ τ ∈ X | v_t(s, τ) ≥ 1 - α }.
"""


function var(v::Array{Float64, 3},
             X::Vector{Float64},
             t::Int,
             s::Int,
             α::Float64)

    K      = length(X)
    vals   = view(v, t+1, s, :)  # [v_t(s, X[1]), ..., v_t(s, X[K])]
    target = 1.0 - α

    # I'm excluding +/-Inf from the VaR search
    first_idx = (X[1]   == -Inf) ? 2     : 1
    last_idx  = (X[end] ==  Inf) ? K - 1 : K

    idx = nothing
    for k ∈ 1:K
        if vals[k] ≥ target
            idx = k
        end
    end

    return idx === nothing ? nothing : X[idx]
end


end # module VarDP
