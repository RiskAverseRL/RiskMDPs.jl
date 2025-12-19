module VarDP2

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
    @assert i ≥ 1 
    return i
end

@inline function hp(mesh::Mesh, x::Float64)
    i = searchsortedfirst(mesh.xs, x)    # smallest index i with xs[i] ≥ x
    @assert i ≤ length(mesh.xs)
    return i
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
    vi(mdp, X, T; side = :plus)

value iteration on an IntMDP with a x-grid that may change with time.
Here, t is the backward DP time index (time-to-go).
Basically, t = how many steps of reward accumulation have occurred from the terminal condition.
so: t=0 → terminal CDF / base case (no)
    t=1 → after 1 step of reward accumulation
    ...
    t=T → after T steps of reward accumulation
Arguments:
    mdp     :: IntMDP
    X  :: Vector{Vector{Float64}}   # X[t+1] is grid for v_t
    T       :: Int                       # time horizon
    side    :: Symbol                    # :plus (h^+) or :minus (h^-)

Returns:
    V :: Vector{Matrix{Float64}} of length T+1
         V[t+1] is an S×K_t matrix with V[t+1][s,k] = v_t(s, X[t+1][k])

    π :: Vector{Matrix{Int}} of length T
         π[t] is an S×K_t matrix with π[t][s,k] = greedy action at time t
"""

function vi(mdp::IntMDP,
            X::Vector{Vector{Float64}},
            T::Int;
            side::Symbol = :plus)

    S = state_count(mdp)
    @assert length(X) == T + 1  # need grids for t = 0,...,T

    # Storage: V[t+1] is S×K_t matrix for v_t
    V = Vector{Matrix{Float64}}(undef, T + 1)
    π = Vector{Matrix{Int}}(undef, T)

    # t = 0: initialize v_0 on X[1]
    X0 = X[1]
    K0 = length(X0)
    V[1] = zeros(Float64, S, K0)

    # t = 0 : V_0 (s,x) = 1{x ≥ 0}
    for s in 1:S, k in 1:K0
        V[1][s, k] = X0[k] ≥ 0.0 ? 1.0 : 0.0
    end

    # Backward recursion: compute v_t from v_{t-1}
    for t in 1:T
        X_cur = X[t+1]    # grid for V[t+1] = v_t
        K = length(X_cur)

        V[t+1] = zeros(Float64, S, K)
        π[t]   = zeros(Int,     S, K)

        mesh_t = Mesh(X[t])      # mesh corresponding to V[t]

        # h^- and h^+ index functions for V[t]
        hm_idx(x::Float64) = hm(mesh_t, x)
        hp_idx(x::Float64) = hp(mesh_t, x)

        for s in 1:S, k in 1:K
            x = X_cur[k]

            max_val = -Inf
            best_a  = first(actions(mdp, s))

            for a in actions(mdp, s)
                nxt = getnext(mdp, s, a)
                states_ns = nxt.states
                probs     = nxt.probabilities
                rewards   = nxt.rewards

                exp_val = 0.0
                @inbounds for i in eachindex(states_ns, probs, rewards)
                    ns = states_ns[i]
                    p  = probs[i]
                    p == 0.0 && continue

                    r_sa = rewards[i]
                    nx   = x + r_sa  # next x value
                    j    = (side == :plus) ? hp_idx(nx) : hm_idx(nx)

                    exp_val += p * V[t][ns, j]
                end

                if exp_val > max_val
                    max_val = exp_val
                    best_a  = a
                end
            end

            V[t+1][s, k] = max_val
            π[t][s,  k]  = best_a
        end
    end

    return V, π
end

"""
    var(V, X, t, s, α)

VaR for time varying meshes:

  
  τ_t(s) = max{τ ∈ X[t+1] | v_t(s, τ) ≥ 1 - α }.

Here V[t+1] is the S×K_t matrix for v_t.
"""
function var(V::Vector{Matrix{Float64}},
            X::Vector{Vector{Float64}},
            t::Int,
            s::Int,
            α::Float64)

   
    X = X[t+1]
    vals = @view V[t+1][s, :]   # row for state s
    # K = length(X)

    target = 1.0 - α

 
    # idx = nothing
    # for k ∈ 1:K
    for k ∈ eachindex(X)
        if vals[k] ≥ target #(v[x1],v[x2],...,v[xk])
            # idx = k
            return X[k]
        end
    end
    return nothing
    # return idx ===nothing ? nothing : X[idx]
end


end # module