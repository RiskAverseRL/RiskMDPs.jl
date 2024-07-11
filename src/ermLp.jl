import MDPs: qvalue
using Revise
using MDPs
using LinearAlgebra
using JuMP


# ---------------------------------------------------------------
# ERM with the total reward criterion.
# ---------------------------------------------------------------
# Needs a terminal state that is a sink and has a reward 0
# Corresponds to an indefinite horizon

"""
Represents an ERM objective with a total reward objective over
an infinite horizon. This formulation is roughly equivalent 
to using a discount factor of 1.0

"""
struct InfiniteERM <: StationaryDet
    β::Float64  # risk level

    function InfiniteERM(β::Number)
        β ≥ zero(β) || error("Risk β must be non-negative")
        new(β)
    end
end

