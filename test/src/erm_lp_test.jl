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
function evaluate_policy(model::TabMDP, π::Vector{Int}, β::Real)
    # evaluation helper variables
    episodes = 500
    horizon::Integer = 100
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    # for the uniform initial state distribution, call each non-sink state equal times
    erm_ave = 0.0 
    states_number = state_count(model)
    returns = []
    for inistate in 1: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)
        #@infiltrate

        # rets is a vector of the total rewards, size of rets = number of episodes
        rets = rweights' * H.rewards |> vec
        
        ret_erm = ERM(rets, ones(length(rets)) / length(rets), β)
        erm_ave += ret_erm * 1.0/(states_number -1)
    end
   
    println("Simulated ERM return: ", erm_ave)
    returns
end


# evaluate_sim_plot is different from evaluate_sim. The function evaluate_sim_plot
# exclude the the first state 1 (no money). The initial capital is at least 1 (state 2).
function evaluate_policy_plot(model::TabMDP, π::Vector{Int})

    # evaluation helper variables
    episodes = 500
    horizon::Integer = 50
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    # for the uniform initial state distribution
    states_number = state_count(model)
    returns = []
    # inistate = 1, no money; 
    for inistate in 2: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)
        #@infiltrate

        # rets is a vector of the total rewards, size of rets = number of episodes
        rets = rweights' * H.rewards |> vec
        
        # returns is an array of returns for all episodes
        for r in rets
            push!(returns,r) 
        end
    end  
    returns
end


# Histogram for single policy
function  plot_histogram_float(returns, α)

    max_value = 9.0
    min_value = -1.0
    b_range = range(-1, 9, length=11)

    p = histogram(returns,bins = b_range,normalize=:pdf,legend = false,xticks = (-1:9,-1:9),
                yticks = 0:0.1:1,bar_width = 0.5)
    xlims!(min_value, max_value)
    ylabel!("Relative frequency")
    xlabel!("Final capital")
    title!("α = $(α)")
    savefig(p,"capital_$α.pdf")

end

# Histogram for three polices
function plot_histogram_multiple(returns_plot_1, α1,returns_plot_3, α3,returns_plot_7, α7,win_p )
        
    max_value = 9.0
    min_value = -1.0
    b_range = range(-1, 9, length=11)

    p1 = histogram(returns_plot_1,bins = b_range,normalize=:pdf,legend = false,xticks = (-1:9,-1:9),
                yticks = 0:0.1:1,bar_width = 0.5)
    xlims!(min_value, max_value)
    ylabel!("Relative frequency")
    xlabel!("Final capital")
    title!("α = $(α1)")

    p3 = histogram(returns_plot_3,bins = b_range,normalize=:pdf,legend = false,xticks = (-1:9,-1:9),
    yticks = 0:0.1:1,bar_width = 0.5)
    xlims!(min_value, max_value)
    ylabel!("Relative frequency")
    xlabel!("Final capital")
    title!("α = $(α3)")

    p7 = histogram(returns_plot_7,bins = b_range,normalize=:pdf,legend = false,xticks = (-1:9,-1:9),
    yticks = 0:0.1:1,bar_width = 0.5)
    xlims!(min_value, max_value)
    ylabel!("Relative frequency")
    xlabel!("Final capital")
    title!("α = $(α7)")

    p = plot(p1, p3, p7, layout=(1, 3), legend=false)
    savefig(p,"capitals$win_p.pdf")
    
end

function main()

    # The values below are for mg0.8.csv
    α1 = 0.1
    π1 ::Vector{Int} = [1,3,4,5,6,7,8,9,1,1]
    β1 = 1.9982277118

    α3 = 0.3
    π3 ::Vector{Int} = [1,3,2,2,2,2,2,2,1,1]
    β3 = 0.83628965110

    α7 = 0.7
    π7 ::Vector{Int} = [1,2,2,2,2,2,2,2,1,1]
    β7 =  0.3760661166107653

    #--------
    # mg0.85.csv, mg0.75.csv, mg0.8.csv, win_p = 0.85, 0.75, or 0.8
    # The variable win_p represents the probability of winning one game.
    #--------
    win_p = 0.8
    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                    "data", "mg$win_p.csv")
    
    # Single state, for unbounded ERM plot with TRC
    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                 "data", "single_tra.csv")

    model = load_mdp(File(filepath))

    #--------
    # Evaluate the optimal policy and simulate the ERM value functions 
    # β will be replaced by β1 or β3 or β7
    #--------
    # returns = evaluate_policy(model, π, β)


    #--------
    #  plot for one tuple (α,π,β),  β will be replaced by β1 or β3 or β7
    #  π will be replaced by π1 or π3 or π7; α  will be replaced by α1 or α3 or α7
    #------
    # returns_plot = evaluate_policy_plot(model, π, β)
    # plot_histogram_float(returns_plot, α )

        
    # plot for three tuples of (α,π,β)
    returns_plot_1 = evaluate_policy_plot(model, π1)
    returns_plot_3 = evaluate_policy_plot(model, π3)
    returns_plot_7 = evaluate_policy_plot(model, π7)
    plot_histogram_multiple(returns_plot_1, α1,returns_plot_3, α3,returns_plot_7, α7 ,win_p)

end

main()
