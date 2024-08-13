using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using MDPs
using JuMP, HiGHS
using CSV: File
using RiskMDPs
using Plots
#using PlotlyJS
using Infiltrator
using CSV


# ---------------------------------------------------------------
# ERM with the total reward criterion. an infinite horizon. This formulation is roughly equivalent 
# to using a discount factor of 1.0. The last state is considered as the sink state
# 1) compute the optimal policy for EVaR and ERM
# 2) Plot for unbounded ERM values
# 3) plot the optimal polices
# 4) The plot is saved .\RiskMDP.jl
# ---------------------------------------------------------------

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


function evar_discretize_beta(α::Real, δ::Real)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 2e-7
    #β1 = 0.001 # for the plot of single state, unbounded erm
    βK = -log(α) / δ

    βs = Vector{Float64}([])
    β = β1

    # experimenting on small β values
    β_bound = minimum([10,βK]) 
    while β < β_bound
        append!(βs, β)
        β *= log(α) / (β*δ + log(α)) *1.05
        # for the plot of single state, unbounded erm 
        #β *= log(α) / (β*δ + log(α))  

    end

    βs

end

"""
Compute B[s,s',a],  b_s^d, B_{s,s'}^d, d_a(s), assume the decsion rule d is deterministic,that is,
 d_a(s) is always 1. 
 a is the action taken in state s
when sn is the sink state, then B[s,a,sn] =  b_s^d, 
when sn is a non-sink state,   B[s,a,sn] = B_{s,s'}^d.
"""
function compute_B(model::TabMDP,β::Real)
    
    states_size = state_count(model)
    actions_size = maximum([action_count(model,s) for s in 1:states_size])

    B = zeros(Float64 , states_size, actions_size,states_size)
     
    for s in 1: states_size
        action_number = action_count(model,s)
        for a in 1: action_number
            snext = transition(model,s,a)
            for (sn, p, r) in snext
                B[s,a,sn] += p  * exp(-β * r) 
            end
        end
    end 
    B
end

"""
Linear program to compute erm exponential value function w and the optimal policy
Assume that the last state is the sink state
"""
function erm_linear_program(model::TabMDP, B::Array, β::Real)
    # TODO: this assumes that the last state is the sink!!
    # TODO: this function should probably not take both β and B because
    # they may not be consistent and it is almost impossible to check
    # whether they are

    # β is only used to compute regular value function v from w
    
     state_number = state_count(model)
     π = zeros(Int , state_number)
     constraints::Vector{Vector{ConstraintRef}} = []

     lpm = Model(HiGHS.Optimizer)
     set_silent(lpm)
     @variable(lpm, w[1:(state_number-1)] )
     @objective(lpm, Min, sum(w))

    #constraints for non-sink states and all available actions
    for s in 1: state_number-1
        action_number = action_count(model,s)
        c_s::Vector{ConstraintRef} = []
        for a in 1: action_number
            push!(c_s, @constraint(lpm, w[s] ≥ B[s,a,1:(state_number-1)] ⋅ w -
                      B[s,a,state_number] ))
        end
        push!(constraints, c_s)
    end

    optimize!(lpm)

    # Check if the linear program has a feasible solution 
    if termination_status(lpm) ==  DUAL_INFEASIBLE || w[(state_number - 1) ] == 0.0
        return  (status = "infeasible", w=zeros(state_number),v=zeros(state_number),π=zeros(Int64,state_number))
    else
         # Exponential value functions
         w = vcat(value.(w), [-1.0])

         #Regular value functions 
         v = -inv(β) * log.(-value.(w) )

         # Check active constraints to obtain the optimal policy
         π = vcat(map(x->argmax(dual.(x)), constraints), [1])
         #println(map(x->dual.(x), constraints))
       
        return (status ="feasible", w=w,v=v,π=π)
    end 
end


# Given different α values, compute the optimal policy for EVaR and ERM
function compute_optimal_policy(alpha_array,initial_state_pro, model,δ)
    
    #Save erm values and beta values for plotting unbounded erm 
    erm_values = []
    beta_values =[]
    opt_policy = []

    for α in alpha_array
        βs =  evar_discretize_beta(α, δ)
        max_h =-Inf
        optimal_policy = []
        optimal_beta = -1
        optimal_v = []
        optimal_w = []

        for β in βs
            B = compute_B(model,β)
            status,w,v,π = erm_linear_program(model,B,β)
            
            # compute the feasible and optimal solution
            if cmp(status,"infeasible") ==0 
                break
            end

            # Calculate erm value. result is one number
            erm = compute_erm(v,initial_state_pro, β)

            # Save erm values and β values for plots
            append!(erm_values,erm)
            append!(beta_values,β)

            # Compute h(β) 
            h = erm + log(α)/β

            if h  > max_h
                max_h = h
                optimal_policy = deepcopy(π)
                optimal_beta = β
                optimal_v = deepcopy(v)
                optimal_w = deepcopy(w)
            end
        end

        opt_erm = max_h - log(α)/optimal_beta

        println("α = ", α)
        println(" EVaR =  ", max_h  )
        println(" π* =  ", optimal_policy)
        println(" β* =  ", optimal_beta)
        println("ERM* = ",opt_erm)
        #println(" vector of regular erm value is  ",optimal_v)
        #println(" vector of exponential erm value is  ",optimal_w)
        push!(opt_policy,optimal_policy)

    end
    (erm_values,beta_values,opt_policy)
end

# Compute a single ERM value using the vector of regular value function and initial distribution
function compute_erm(value_function :: Vector, initial_state_pro :: Vector, β::Real)

    # -1.0/β * log(∑ μ_s * exp())
    sum_exp = 0.0
    for index in 1:length(value_function)
        sum_exp += initial_state_pro[index] * exp(-β * value_function[index])
    end

    result = -inv(β) * log(sum_exp)

    result
end


# Show unbounded ERM value functions, NO simulation for Infeasible solutions 
# plot erm values for a discounted MDP and a transient MDP, single state
function  erms_dis_trc(erm_trc, betas)
    
    erm_discounted = -2
    erm_dis = fill(erm_discounted,size(betas))

    p=plot(betas,erm_trc,label="trc", linewidth=3,legend = :outertopright)
    plot!(betas,erm_dis,label="discounted", linewidth=3)
    xlims!(minimum(betas),last(betas))
    ylims!(-12.5,-1.5)
    xlabel!("β")
    ylabel!("ERM value function")
    savefig(p,"erm_values_unbounded.pdf")

 end

 # Given four different α values, plot the optimal policies
 function plot_optimal_policy()

# Delete the first state, last state and the sink state
# investment = action -1
α1 = 0.9
π1 = [1, 2, 2, 2, 4, 3, 2, 1, 1]
y1 =[1, 1, 1, 3, 2, 1]

α2 = 0.7
π2 = [1, 2, 2, 2, 2, 2, 2, 1, 1]
y2 =[1, 1, 1, 1, 1, 1,]


α3 = 0.40
π3 = [1, 3, 2, 2, 2, 2, 2, 1, 1]
#y3 = [3, 2, 2, 2, 2, 2]
y3 = [0, 1, 1, 1, 1, 1] # 3, quit, draw 0


α4 = 0.20
π4 = [1, 3, 4, 5, 6, 7, 8, 1, 1]
#y4 = [ 3, 4, 5, 6, 7, 8]
y4 = [ 0, 0, 0, 0, 0, 0] # quit, draw 0

   
   states= []
   for index in 1:6
       push!(states,index)
   end

   ymax = 4

    p = scatter(states, y1,  markershape = :star5,markersize = 8,markeralpha = 0.6,
                markercolor=RGBA(1, 1, 1, 0),label = "α = $α1",markerstrokecolor = "blue",
                 markerstrokewidth=3.5,xticks = 1:1:7, yticks = 0:1:ymax,
                 legend=:topleft)

        scatter!(states, y2;  markersize=11,markershape=:rect, markeralpha = 0.6,
                 markercolor=RGBA(1, 1, 1, 0),label = "α = $α2",markerstrokecolor = "darkgreen",
                 markerstrokewidth=3.5, xticks = 1:1:7, yticks = 0:1:ymax )
                 
        scatter!(states, y3,  markershape = :hexagon ,markersize = 28,markeralpha = 0.6,
                markercolor=RGBA(1, 1, 1, 0), label = "α = $α3",markerstrokecolor = "red",
                markerstrokewidth=3.5,xticks = 1:1:7, yticks = 0:1:ymax )

        scatter!(states, y4,  markershape = :circle,markersize = 22,markeralpha = 0.6,
                 markercolor=RGBA(1, 1, 1, 0),label = "α = $α4",markerstrokecolor = "purple",
                markerstrokewidth=3.5,xticks = 0:1:7, yticks = 0:1:ymax )

            
         xlims!(0,7)
         ylims!(0,4)
         xlabel!("state")
         ylabel!("optimal action")
    
         savefig(p,"policy.pdf")

 end


function main_evar()

    δ = 0.01

    # # file for a single state, plot unbunded erm and constant erm 
    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                    "data", "single_tra.csv")
    # initial_state_pro = [1,0]
    # alpha_array = [0.7]


    # 7 mgp0.68.csv; 0.68 is the probabilty of winning a game
    # Gambler(7,0.68)

    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                   "data", "7 mgp0.68.csv") 
                    
    model = load_mdp(File(filepath))

    # For the gambler domain
    # Set 0 to the initial probability of the first state
    state_number = state_count(model)
    initial_state_pro = Vector{Float64}()
    append!(initial_state_pro,0) 
    for index in 2:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-1)) 
    end
    # add the sink state with the initial probability 0
    append!(initial_state_pro,0) 
    
    # risk level of EVaR
     alpha_array = [0.2,0.4,0.7,0.9]


    #Compute the optimal policy 
    erm_trc, betas,opt_policy = compute_optimal_policy(alpha_array,initial_state_pro, model,δ)


    # plot erm values in a discounted MDP and a transient MDP
    #erms_dis_trc(erm_trc, betas)


    # Plot the optimal policies. The optimal polices are copied inside the function
    #plot_optimal_policy()


    # Value iteration
    #   v,_,_ = value_iteration(model, InfiniteH(0.7); iterations = 10000)
    #   println("---------------------------")
    #   println("expectation, value functions: ",sum(v .* initial_state_pro))
  
end 

########

function main_erm()

    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                "data", "15 mgp0.8.csv")
    
    filepath = joinpath(dirname(pathof(RiskMDPs)), 
    "data", "7 mgp0.68.csv")

    model = load_mdp(File(filepath))
    
    state_number = state_count(model)
    initial_state_pro = Vector{Float64}()
    for index in 1:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-1)) # start with a non-sink state
    end
    append!(initial_state_pro,0) # add the sink state with the initial probability 0

    
    β_array = [ 1.1e-7,1.2e-7,1.3e-7,1.4e-7,1.5e-7,1.6e-7,1.7e-7,1.8e-7,1.9e-7]
    #β_array = [ 1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    
    for β ∈ β_array
        B = compute_B(model, β)
        status,w,v,π = erm_linear_program(model, B, β)
        println("+++++++++++++++++ ", β)
        #println("B = \n $")
        #println("-----------------")
        #println("w =\n", w)
        #println("-----------------")
       # println("v =\n", v)
        println("π =\n", π)
        println("-----------------")
    end
end 

####

main_evar()
#main_erm()







