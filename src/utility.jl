using LinearAlgebra
using MDPs

# others
import Base: length

# *****************************************************************
# IMPLEMENTATION NOTE: The algorithms do not work by simply augmenting the states 
# because that makes it difficult to generalize to
# value function approximation, particularly over the augmented state space
# *****************************************************************


# --------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------

"""
Objective that is solved using a deterministic policy
which is Markov on an augmented state space
"""
abstract type AugmentedMarkov <: Markov end


"""
Represents the utility function u(t,z) with a parameter t in R
"""
abstract type Utility end

struct UtilityVaR <: Utility end
struct UtilityCVaR <: Utility end
struct UtilityEVaR <: Utility end


"""
    AugUtility(γ, α, T)

Represents an objective for a utility function u(t,z) where t 
the state space and assumes a discount factor `γ`.

The value function and the q functions are represented as a collection of values
for each of the risk levels η.
"""
struct AugUtility{U} <: AugmentedMarkov where {U <: Utility}
    γ::Float64  # discount factor
    u::U        # utility function
    T::Int      # horizon
    mesh::Vector{Float64}  # absolute mesh (the same across time horizon)

    function AugUtility(γ::Number, u::U, T::Integer,
                        mesh::Vector{Float64}) where {U <: Utility}
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        issorted(mesh) || error("Mesh values must be sorted increasingly")
        new{U}(γ, u, T, mesh)
    end
end

# -----------------------------------------------------------------------
# TODO: make the mesh relative to the scale of the value function
#       perhaps it would be good to use quantiles to represent
#       mesh points; the problem may be the computational complexiy
#       of computing the quantiles
# -----------------------------------------------------------------------

"""
    value_mesh(mdp, au, t)

Return the discretization for the objective `au` and mdp `mdp` for the value
function at time `t`
"""
value_mesh(mdp::TabMDP, au::AugUtility, t::Int) = au.mesh

# -----------------------------------------------------------------------
# Discretization
# -----------------------------------------------------------------------

"""
Represents a sorted vector of floats. Used to eliminate excessive checks
for sortedness when constructing `Discretized` from existing meshes
"""
struct SortedVector
    x :: Float64
    function SortedVector(x::Vector{Float64})
        issorted(x) || error("mesh values must be sorted")
        new(x)
    end
end

"""
Represents a discretized value for a mesh. It could represent the value
function or a policy. The `mesh` is the x value and `values` is a y value.
"""
struct Discretized{T}
    mesh :: SortedVector
    values :: Vector{T}

    function Discretized{T}(mesh::SortedVector, values::Vector{T}) where {T}
        length(mesh) == length(values) || error("Lengths must be the same")
        new{T}(mesh, values)
    end

    function Discretized{T}(mesh::Vector{Float64}, values::Vector{T}) where {T}
        length(mesh) == length(values) || error("Lengths must be the same")
        new{T}(mesh, SortedVector(values))
    end
end

length(d::Discretized) = length(d.mesh)


""" Augmented value function """
const DscValue = Discretized{Float64}
""" Augmented policy values (actions) """
const DscAction = Discretized{Int}
#""" Augmented transition probabilities to the next mesh value """
#const DscTransition = Discretized{Int}

# return the parameters and values for a discretized value function
## params(d::Discretized) = d.mesh
## values(d::Discretized) = d.values

"""
    pw_const_near(d, x)

Interpret the discretized values in `d` as piecewise constant with
the value of the function being the closest mesh element.
"""
function pw_const_near(d::Discretized, x::Real)
    xlast = searchsortedlast(d.mesh, x)
    # element smaller than the range
    xlast < 1 && return first(d.values)

    xfirst = searchsortedfirst(d.mesh, x)
    xfirst > length(d.mesh) && return last(d.values)

    if abs(d.mesh[xfirst] - x) < abs(d.mesh[xlast] - x)
        return d.values[xfirst]
    else
        return d.values[xlast]
    end
end

## TODO: also implement a left and right continuous values

# --------------------------------------------------------------------
# Value function definition
# --------------------------------------------------------------------

const AuValue = Vector{DscValue}
const AuPolicy = Vector{DscValue}
#const AuTransition = Vector{DscTransition}

"""
    constant_value(mdp, objective, value)

Create a constant value function for the augment objective for
all states in the MDP
"""
function constant_value(model::TabMDP, objective::AugUtility,
                        value::Float64, t::Int)
    mesh = value_mesh(model, objective, t)
    # TODO: is there a better way to do this?
    [DscValue(mesh, fill(value, length(mesh))) for _ in 1:state_count(model)]
end


# --------------------------------------------------------------------
# Transition functions for targets
# --------------------------------------------------------------------

# tabular transition
const TabTransition = Transition{Int, Int}

"""
    next_au_target(obj, target, tran)

Return the next target level (augmentation) for an objective `obj` when starting in
`target` and transitioning according to the `Transition` variable `tran`.
"""
function next_au_target end

next_au_target(obj::AugUtility{UtilityVaR}, target::Real, tran::Transition) =
    (Float64(target) - tran.reward) / obj.γ :: Float64

"""
   au_reward(obj, target, tran)

Return reward attained for an objective `obj` when starting in
target `target`.
"""
function au_reward end

au_reward(::AugUtility{UtilityVaR}, ::Real, ::Transition) = 0.0 :: Float64


"""
   au_discount(obj, g, tran)

Return reward attained for an objective `obj` when starting in
target `g`.
"""
function au_discount end

au_discount(::AugUtility{UtilityVaR}, ::Real, ::Transition) = 1.0 :: Float64

"""
    au_terminal_value(obj, g, tran)

Compute terminal value function for an objective `obj` and a target
level `g`
"""
function au_terminal_value end

au_terminal_value(obj::AugUtility{UtilityVaR}, g::Real) = 
    (g ≤ zero(g) ? 1.0 : 0.0) :: Float64

# --------------------------------------------------------------------
# Bellman update functions
# --------------------------------------------------------------------


"""
    qvalue(mdp, [t], obj, s, a, v)

Compute the qfunction for the augmented utility MDP objective with the 
state `s` and action `a`. Also computes the target levels for the next state,
which is useful when deploying a policy.

The function uses a piecewise constant discretization of the utility value function
"""
function qvalue end 

# just translates the time-dependent update to a time-independent one
# the second parameter is t = time
qvalue(mdp::MDP{S,A}, _::Int, obj::AugUtility, s::S, a::A, v) where {S,A} =
    qvalue(mdp, obj, s, a, v)


# compute the qvalue for a given state and action
function qvalue(mdp::TabMDP, obj::AugUtility, s::Int, a::Int, v::AuValue)
    mesh = value_mesh(obj)
    n = length(mesh)

    qvalues = zeros(n) 
    # iterate over target levels g
    for (meshind,target) ∈ enumerate(mesh)
        # iterate over all next states
        for (s′, p, r) ∈ transition(mdp, s, a)
            # TODO: make sure that the things do not break even when a single
            #  s′ is repeated multiple times with different rewards
            tran = Transition(s, a, r, s′, -1)
            g′ = next_au_target(obj, target, tran)
            qvalues[meshind] += p * (au_reward(obj, target, tran) +
                au_discount(obj, target, tran) * pw_const_near(v[s′], g′))
        end
    end
    DscValue(mesh, qvalues)
end

bellmangreedy(mdp::MDP{S,A}, t, obj::AugUtility, s::S, v) where {S,A} =
    bellmangreedy(mdp, obj, s, v )
    
# compute the maximum for each discretization level
function bellmangreedy(mdp::MDP{S,A}, obj::AugUtility, s::S,
                       v::AuValue) where {S,A}

    isterminal(mdp, s) && error("Terminal states not supported")

    mesh = value_mesh(obj)

    acts = actions(mdp, s)
    qvalues = [qvalue(mdp, obj, s, a, v) for a ∈ acts]
    
    # holds results
    bestactions = Vector{Int}(undef, length(mesh))
    valuefunctions = Vector{Float64}(undef, length(mesh))

    # compute the best action for each one of the risk levels
    for meshind ∈ eachindex(mesh)
        amax = argmax(qvalues[a].values[meshind] for a ∈ eachindex(acts))
        bestactions[meshind] =  acts[amax]
        valuefunctions[meshind] = qvalues[amax].values[meshind]
    end

    (qvalue = DscValue(obj, valuefunctions), 
        action = DscAction(obj, bestactions))
end

# --------------------------------------------------------------------
# Value iteration
# --------------------------------------------------------------------

"""
    value_iteration(model, objective; [vterminal]) 

Compute value function and policy for a tabular MDP `model` with an augmented objective
an objective `objective`. The time steps go from 1 to T+1, the last decision happens
at time T.

The method assumes that all reward beyond the terminal time `objective.T` is 0 and
the value function must be initialized by `au_terminal_value`
"""
function value_iteration end

function value_iteration(model::TabMDP, objective::AugUtility)
    n = state_count(model)

    # the outer array is over the time steps
    v = Vector{AuValue}(undef, horizon(objective)+1)
    π = Vector{AuPolicy}(undef, horizon(objective))
    
    # initialize final value function (the values are the same for all states)
    endvalue = [au_terminal_value(objective, g) for g in objective.mesh]
    v[end] = fill(endvalue, n) # references to the same value intentional

    for t ∈ horizon(objective):-1:1
        # initialize vectors
        v[t] = Vector{DscValue}(undef, n)
        π[t] = Vector{DscAction}(undef, n)
        Threads.@threads for s ∈ 1:n           
            local bg = bellmangreedy(model, t, objective, s, v[t+1])
            v[t][s] = bg.qvalue
            π[t][s] = bg.action
        end
    end
    return (policy = π, value = v)
end

function value_iteration(mdp::TabMDP, objective::AugmentedMarkov)
    value_iteration(mdp, objective, constant_value(mdp, objective, 0.))
end

# TODO: implement methods that compute upper and lower bounds on the value function
# then it is possible to estimate the error in the approximation; this is
# similar to the lower and upper bounds computed in our EVaR approximation


# --------------------------------------------------------------------
# Risk measure computation (risk from utility)
# --------------------------------------------------------------------

"""
    utility_risk(obj, value, α; [ubound = Inf64])

Compute the corresponding initial target from the value function. Typically,
`value` would be the value function in the initial state and the first time
step (=1). The risk level is `α`.

The result is the initial target value and the initial estimated value. These
values are useful in simulation and when coputing the actual predicted risk value.
"""
function utility_risk end

function utility_risk(::AugUtility{UtilityVaR}, targets::DscValue, α::Real)  
    @assert issorted(targets.values)
    # TODO: this method really needs test cases

    # the VaR definition for a level α is
    # sup { t | P[X >= t] ≥ 1-α } = sup { t | P[X <= t] ≤ α }
    # = inf { t | P[X <= t] > α }

    # finds the largest value ≤ α and add 1 to get the supremum
    index = searchsortedlast(targets.values, α) + 1
    # check if it is infinite
    if index > length(targets.values)
        return Inf64
    else
        return targets.values[index]
    end
end

# TODO: This definition of VaR is not consistent with the definition
# in RiskMeasures, the α here is 1-α there

# --------------------------------------------------------------------
# Simulation policy (history-dependent and time-dependent)
# --------------------------------------------------------------------

const AuState = NamedTuple{(:time, :target), Tuple{Int64, Float64}}

"""
    PolicyAugU(obj, policy, target_start)

Policy with the internal state that represents the time step and the
current target value.
"""
struct PolicyAugU{U} <: Policy{Int,Int,AuState} where {U <: Utility}
    obj :: AugUtility{U}          # objective
    policy :: Vector{AuPolicy}    # time, state, target
    target_start :: Float64       # initial target value
end

# creates the internal simulation state
function make_internal(::TabMDP, π::PolicyAugU{U},
                       ::Int) :: AuState where {U <: Utility}
    # find the smallest risk index that is greater than 
    (time = 1, target = π.target_start)
end

function append_history(π::PolicyAugU, internal::AuState, tran::Transition) :: AuState
    @assert tr.time == internal.time + 1 # advances by one
    # update the risk index
    next_t = next_au_target(π.obj, internal.target, tran)
    # make sure the state is actually found
    (time = tr.time, target = next_t)
end

# return the best action for the closest known point
take_action(π::PolicyAugU, int::AuState, s::Int) = 
    pw_const_near(π.policy[int.time][s], int.target)
