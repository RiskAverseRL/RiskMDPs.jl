"""
Simulation for linear program of erm and evar with a transient MDP
"""
using DataFrames: DataFrame
using DataFramesMeta
using MDPs
using CSV: File
using RiskMDPs
using RiskMeasures

struct Problem{M <: TabMDP}
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    ΔR::Float64
    horizon::Int
    initstate::Int
    model::M
end

function make_domains()
    problems = Dict{String, Problem}()
      # ruin
      begin
        # risk parameters
        α = 0.9           # var, cvar, evar
        β = 0.5            # erm
        γ = 0.95
        δ = 0.5            # discretization error
        ΔR = 1. / (1-γ)
        horizon = 200
        initstate = 8  # capital: state - 1
        model = Gambler.Ruin(0.7, 10)
        problems["ruin"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    return problems
end

"""
load mdp from a csv file, 1-based index
"""
function load_mdp(input)
    mdp = DataFrame(input)
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)
    
    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{IntState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{IntAction}(undef, maximum(sd.idaction))
       
        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            try 
            actions[idaction] = IntAction(ad.idstateto, ad.probability, ad.reward)
            catch e
                error("Error in state $(idstate-1), action $(idaction-1): $e")
            end
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                " that were uninitialized " * string(findall(.!action_init) .- 1 ) ))

        states[idstate] = IntState(actions)
        state_init[idstate] = true
    end
    IntMDP(states)
end

function evar_discretize2(α::Real, δ::Real, ΔR::Number)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 8*δ / ΔR^2
    βK = -log(1-α) / δ

    βs = Vector{Float64}([])
    β = β1
    while β < βK
        append!(βs, β)
        β *= log(1-α) / (β*δ + log(1-α))
    end
    
    βs
end

function evaluate_sim(model::TabMDP, π :: Vector, horizon::Real, initstate::Vector, α::Real)

    roundresult(x) = round(x; sigdigits = 3)

    episodes = 10000

    # evaluation helper variables
    rweights::Vector{Float64} = 1 .^ (0:prob.horizon-1)     # reward weights
    edist::Vector{Float64} = ones(episodes) / episodes # distribution over episodes

    H = simulate(model, π, initstate, horizon, episodes)
    returns = rweights' * H.rewards |> vec

    solevar = evar(returns, edist, α) 
    revar = solevar.evar |> roundresult
    println("Evaluation EVaR opt β ", solevar.β, ", EVaR ", revar)

end

function evaluate_sim(prob::Problem, π::Vector)

    # evaluation helper variables
    episodes = 10000
    rweights::Vector{Float64} = prob.γ .^ (0:prob.horizon-1)     # reward weights
    edist::Vector{Float64} = ones(episodes) / episodes # distribution over episodes
    
    H = simulate(prob.model, π, prob.initstate, prob.horizon, episodes)
    returns = rweights' * H.rewards |> vec
    return sum(returns) / length(returns)
end


function main()

α = 0.8 # risk level of EVaR
δ = 0.1
ΔR =1 # how to set ΔR ?? max r - min r: r is the immediate reward

"""
Input: a csv file of a transient MDP, 1-based index
Output:  the model passed in ERM function
"""

filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "ruin.csv_tra.csv")
model = load_mdp(File(filepath))
 βs =  evar_discretize2(α, δ, ΔR)                  
state_number = state_count(model)

π =[1 1 1 1 1 1 1 1 1 1 1 1]

problems = make_domains()
problem = problems["ruin"]
evaluate_sim(problem, π)

intial_state_pro = Vector{Float64}()
for s in 1:state_number-1
    push!(intial_state_pro,1.0/(state_number-1)) # start with a non-sink state
end
push!(intial_state_pro,0) # add the sink state with the initial probability 0
end

main()