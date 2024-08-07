using MDPs
import Base
using Revise
using RiskMDPs
using RiskMeasures
using Distributions
using Accessors
using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using CSV: File
using Infiltrator
using Plots

#----------------------
# 1) Simulate ERM value functions
# 2) plot histogram
#--------------------


"""
load a transient mdp from a csv file, 1-based index
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


####

# evaluates the policy by simulation
function evaluate_sim(model::TabMDP, π::Vector{Int}, β::Real)
    # evaluation helper variables
    episodes = 500
    horizon::Integer = 100
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    # for the uniform initial state distribution, call each non-sink state equal times
    erm_ave = 0.0 
    v_test =[]
    states_number = state_count(model)
    returns = []
    for inistate in 1: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)
        #@infiltrate

        # rets is a vector of the total rewards, size of rets = number of episodes
        rets = rweights' * H.rewards |> vec
        
        # returns is an array of returns for all episodes
        for r in rets
            push!(returns,r)
        end
        
        ret_erm = ERM(rets, ones(length(rets)) / length(rets), β)
        erm_ave += ret_erm * 1.0/(states_number -1)
    end
   
    println("Simulated ERM return: ", erm_ave)
    returns
end


# return can be a float number, gambler domain
# Divide the range of returns into ten bins
function  plot_histogram_float(returns, α)

    max_value = maximum(returns)
    min_value = minimum(returns)
    println("maiximal value is: ", max_value)
    println("minimal value is: ", min_value)
    bin = (max_value - min_value) /10.0

    p = histogram(returns,bins = min_value:bin:max_value,normed=true,legend = false)
    xlims!(min_value, max_value)
    ylabel!("Relative frequency")
    xlabel!("Final capital")
    title!("α = $(α)")
    savefig(p,"capital_$α.pdf")

end

function main()

     
    #π ::Vector{Int} =[1, 1] # For the single state example

    # # The optimal policy, for gm0.1.csv
    # α = 0.1
    # β =  35.628661689092
    # π ::Vector{Int} =   [1, 2, 3, 4, 5, 1, 1]

    α = 0.9
    β =  0.88528658
    π ::Vector{Int} = [1, 2, 2, 2, 2, 1, 1]
   

    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "gm0.1.csv")
                    
    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                 "data", "single_tra.csv")

    model = load_mdp(File(filepath))
    returns = evaluate_sim(model, π, β)


    #  return can be a float number, gambler domain
    plot_histogram_float(returns, α )

end

main()
