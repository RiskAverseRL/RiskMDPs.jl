using DataFrames: DataFrame
using DataFramesMeta
using MDPs
using JuMP, HiGHS
using CSV: File
using RiskMDPs
using Plots

# ---------------------------------------------------------------
# ERM with the total reward criterion. an infinite horizon. This formulation is roughly equivalent 
#to using a discount factor of 1.0. The last state is considered as the sink state
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
function erm_linear_program(model::TabMDP,B::Array,β::Real)

     #lpm = Model(GLPK.Optimizer)
     lpm = Model(HiGHS.Optimizer)
     set_silent(lpm)

     state_number = state_count(model)
     w = zeros(state_number)
     v = zeros(state_number)
     π = zeros(Int , state_number)


     @variable(lpm,w[1: state_number] )
     @objective(lpm,Min,sum(w[1: state_number]))

    constraints::Vector{Vector{ConstraintRef}} = []

    #constraints for non-sink states and all available actions
    for s in 1: state_number-1
        action_number = action_count(model,s)
        c_s::Vector{ConstraintRef} = []
        for a in 1: action_number
            snext = transition(model,s,a)
            # bw is used to save  B_{s,̇̇}^a * \bm{w} in the linear program formulation
            bw = 0 
            for (sn, p, r) in snext
                if sn != state_number # state_number is the sink state
                    bw += B[s,a,sn] *w[sn]
                end
            end
            # constraint for a non-sink state and an action
            push!(c_s,@constraint(lpm, w[s] ≥ -B[s,a,state_number] + bw ))

        end
        push!(constraints, c_s)
    end

    # The constraint for the sink state
    push!(constraints, [@constraint(lpm, w[state_number] == -1)])
    optimize!(lpm)

    # Check if the linear program has a feasible solution
    if termination_status(lpm) ==  DUAL_INFEASIBLE
        return  (status = "infeasible", w=w,v=v,π=π)
    else

         # Exponential value functions
         w = value.(w) 

         #Regular value functions 
         v = -1.0/β * log.(-value.(w) )

         # Initialize a policy and generate an optimal policy
         π = zeros(Int , state_number)

        
         # Check active constraints to obtain the optimal policy
         π = map(x->argmax(dual.(x)), constraints)

        return (status ="feasible", w=w,v=v,π=π)
    end 
end


function evar_discretize_beta(α::Real, δ::Real, ΔR::Number)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 8*δ / ΔR^2
    βK = -log(α) / δ
    println("minimal β =  ",β1 )
    println("maximal β =  ",βK)

    βs = Vector{Float64}([])
    β = β1
    while β < βK
        append!(βs, β)
        β *= log(α) / (β*δ + log(α))
    end
    βs

end

# Compute a single ERM value using the vector of regular value function and initial distribution
function compute_erm(value_function :: Vector, initial_state_pro :: Vector)
    return sum(value_function.*initial_state_pro)
end

# Given different α values, x axis: β; y axis: h(β)
# using PlotlyJS
function h_beta_plot(alpha_array,initial_state_pro, model,δ, ΔR)

    # Initialize arrays to save h(β), β given different α values
    n = size(alpha_array)
    
    beta_array = Vector{Vector{Float64}}(undef,n )
    h_array = Vector{Vector{Float64}}(undef,n)

    d =[1,2]
    #for index in 1:
    for index in d
        beta_array[index] = []
        h_array[index] =[] 
    end

    i = 0
    for α in alpha_array
        βs =  evar_discretize_beta(α, δ, ΔR)
        i += 1
        for β in βs
            B = compute_B(model,β)
            status,w,v,π = erm_linear_program(model,B,β)

            # compute the optimal policy for β that has only feasible solution
            if cmp(status,"infeasible") ==0 
                break
            end

            h = compute_erm(v,initial_state_pro) + log(α)/β
        
            append!(h_array[i],h)
            append!(beta_array[i],β)
        end
    end

    # Extract data for β in the range[0.1,3.0]
    beta_array_t = Vector{Vector{Float64}}(undef,n )
    h_array_t = Vector{Vector{Float64}}(undef,n)
    for index in d
        beta_array_t[index] = []
        h_array_t[index] =[] 
    end

    for i in d
        for (index, β) in pairs(beta_array[i])
            if  β <2
                append!(beta_array_t[i],β)
                append!(h_array_t[i],h_array[i][index])
            end
        end
    end

    p=plot(beta_array_t[1],h_array_t[1],label="α = $(alpha_array[1])", linewidth=3)
    
    d1 =deleteat!(d,1)
    for i in d1
        plot!(beta_array_t[i],h_array_t[i],label="α = $(alpha_array[i])", linewidth=3)
    end

    xlims!(0,2.2)
    ylims!(-2.5,2.5)
    xlabel!("β")
    ylabel!("h(β)")
    savefig(p,"beta_h.pdf")

end


# Compute a single ERM value using the vector of regular value function and initial distribution
function compute_erm(value_function :: Vector, initial_state_pro :: Vector)
    return sum(value_function .* initial_state_pro)
end

# Compute the optimal policy for different α values
function compute_optimal_policy(alpha_array,initial_state_pro, model,δ, ΔR)
    
    #Save erm values and beta values for plotting unbounded erm
    # in transient mdp 
    erm_values = []
    beta_values =[]

    for α in alpha_array
        βs =  evar_discretize_beta(α, δ, ΔR)
        max_h =-Inf
        optimal_policy = []
        optimal_beta = -1
        optimal_v = []


        for β in βs
            B = compute_B(model,β)
            status,w,v,π = erm_linear_program(model,B,β)
            
            # compute the optimal policy for β that has only feasible solution
            if cmp(status,"infeasible") ==0 
                break
            end

            # Calculate erm value. result is one number
            erm = compute_erm(v,initial_state_pro)

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
            end
        end

        opt_erm = max_h - log(α)/optimal_beta

        println("α = ", α)
        println(" EVaR =  ", max_h  )
        println(" π* =  ", optimal_policy)
        println(" β* =  ", optimal_beta)
        println("ERM* = ",opt_erm)
        println(" vector of regular erm value is  ",optimal_v)
    end
    (erm_values,beta_values)
end

# plot erm values for a discounted MDP and a transient MDP, single state
 function  erms_dis_trc(erm_trc, betas, erm_discounted)
    
    erm_dis = fill(erm_discounted,size(betas))
    infeasible_x = []
    infeasible_y =[]
    last_beta = last(betas)
    # Assigne a number to the erm value of infeasible solutions
    trc_infeasible = last(erm_trc)-0.5 

    beta_d = deepcopy(betas)

    #generate data set for infeasible solutions
    x_additonal = range(last_beta, stop=last_beta+0.2, length=200)
    for i in x_additonal
        append!(infeasible_x,i)
        append!(infeasible_y,trc_infeasible)

        # erm values for β greater than the threshold value
        append!(beta_d,i)
        append!(erm_dis,erm_discounted)
    end

    
    p=plot(betas,erm_trc,label="trc", linewidth=3)
    plot!(beta_d,erm_dis,label="discounted", linewidth=3)
    plot!(infeasible_x,infeasible_y,label="infeasible",linewidth=3, ls=:dot)
    xlims!(0,last(infeasible_x))
    ylims!(-10,-1)
    xlabel!("β")
    ylabel!("ERM value function")
    savefig(p,"erm_dis_trc.pdf")

 end

function main()

    δ = 0.01
    ΔR =1 # how to set ΔR ?? max r - min r: r is the immediate reward

    """
    Input: a csv file of a transient MDP, 1-based index
    Output:  the model passed in ERM function
     """
    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                   "data", "g5.csv")
    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                    "data", "single_tra.csv")
                                 
    model = load_mdp(File(filepath))
    
    # Uniform initial state distribution
    state_number = state_count(model)
    initial_state_pro = Vector{Float64}()
    for index in 1:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-1)) # start with a non-sink state
    end
    append!(initial_state_pro,0) # add the sink state with the initial probability 0

    
    # risk level of EVaR
    #alpha_array = [0.15,0.3,0.45,0.6]
    #alpha_array = [0.75,0.85,0.9,0.95]
    alpha_array = [0.85,0.7]

    # plot h(β) vs. β given different α values
    #h_beta_plot(alpha_array,initial_state_pro, model,δ, ΔR)

    #Compute the optimal policy 
    erm_trc, betas =compute_optimal_policy(alpha_array,initial_state_pro, model,δ, ΔR)

   # plot erm values in a discounted MDP and a transient MDP
   #  erm_discounted = r/(1-γ)
    erm_discounted = -2
    #erms_dis_trc(erm_trc, betas, erm_discounted)
  
end 

main()







