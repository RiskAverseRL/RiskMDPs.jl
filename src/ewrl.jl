using CSV
using DataFrames, DataFramesMeta
using CSV: File
using MDPs
using MDPs.Domains
using RiskMDPs
using LinearAlgebra
using JuMP, HiGHS
using Plots
using Infiltrator




# code for the EWRL paper

function constrcut(ϵ::Real)
    state_size = 3 # include the sink state
    action_size = 1
    
    # reward of going from state s to state s’ through action a.
    r = zeros((state_size, action_size,state_size))
    # transition probablity of (state, action, next state, model)
    p = zeros((state_size, action_size,state_size))

    p[1,1,1] = ϵ
    p[1,1,2] = 1- ϵ
    p[2,1,1] = ϵ
    p[2,1,3] = 1 -  ϵ
    p[3,1,3] = 1

    r[1,1,1] = -1
    r[1,1,2] = 1
    r[2,1,1] = -0.5
    r[2,1,3] = 0
    r[3,1,3] = 0

    arr_idstatefrom = Vector{Int}()
    arr_idstateto = Vector{Int}()
    arr_idaction = Vector{Int}()
    arr_prob = Vector{Float64}()
    arr_reward = Vector{Float64}()

    for s in 1:3
        for sn in 1:3
            if p[s,1,sn] > 0.0
                push!(arr_idstatefrom,s)
                push!(arr_idaction, 1)
                push!(arr_idstateto, sn)
                push!(arr_prob,p[s, 1, sn])
                push!(arr_reward, r[s, 1, sn])
            end
        end
    end

    # save the transient MDP
    data = DataFrame(idstatefrom = arr_idstatefrom, idaction = arr_idaction, idstateto = arr_idstateto, probability = arr_prob,
                  reward = arr_reward)
    data
end

"""
load a transient mdp from a csv file, 1-based index
"""
function load_mdp(input)
    #mdp = DataFrame(input)
    mdp = @orderby(input, :idstatefrom, :idaction, :idstateto)
    
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
    #β1 = 2e-7
    β1 = 0.001 # for the plot of single state, unbounded erm
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
    evar = 0.0

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
        evar = max_h

        # println("α = ", α)
        # println(" EVaR =  ", evar  )
        # println(" π* =  ", optimal_policy)
        # println(" β* =  ", optimal_beta)
        # println("ERM* = ",opt_erm)
        #println(" vector of regular erm value is  ",optimal_v)
        #println(" vector of exponential erm value is  ",optimal_w)
        push!(opt_policy,optimal_policy)

    end
    (evar,erm_values,beta_values,opt_policy)
end

# Compute the h(β)
function compute_h(alpha_array,initial_state_pro, model,δ)
    
    #Save erm values and beta values for plotting unbounded erm 
    erm_values = []
    beta_values =[]
    opt_policy = []
    
    hs = []
    optbetas=[]
    evars = []

    optimal_beta = -1

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
            push!(hs,h)

            if h  > max_h
                max_h = h
                optimal_policy = deepcopy(π)
                optimal_beta = β
                optimal_v = deepcopy(v)
                optimal_w = deepcopy(w)
            end

        end
        push!(optbetas,optimal_beta)
        push!(evars, max_h)


    end
    #println(" h values are ", hs)
    (optimal_beta,hs,beta_values,evars,optbetas)
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


 # Given a β value, compute the spectral radius of the policy matrix
 function compute_rho(ϵ::Real, β::Real)

    B = [ϵ*exp(β) (1-ϵ)*exp(-β); ϵ *exp(0.5 * β) 0]
    eigen_value = eigvals(B)
    rho = -Inf
    for x in eigen_value
        if abs(x) > rho
            rho =abs(x)
        end
    end
    rho
 end

 function  evar_plot()

    
    epsilons =[0.89,0.9,0.91]
    evars = []
    # save spectral radius
    δ = 0.01
    
    hss::Vector{Vector{Float64}} = []
    betass::Vector{Vector{Float64}} = []
    evars = []
    optbetas = []

    for ϵ in epsilons

    data = constrcut(ϵ)
    model = load_mdp(data)               
    initial_state_pro = [0.5,0.5,0]

    #risk level of EVaR
    alpha_array = [0.75]

    #Compute the optimal policy 
    opt_beta,hs,betas,evar,optbeta = compute_h(alpha_array,initial_state_pro, model,δ)
    
    push!(hss,hs)
    push!(betass,betas)
    push!(evars,evar)
    push!(optbetas,optbeta)

    end

    p = plot()
    linecolors = [:red, :blue,:darkgreen,:purple]
    linestyles = [:solid,:dash,:dot,:dashdot]
    for i in 1:length(epsilons)
        plot!(betass[i],hss[i], linewidth=1, label = "ϵ = $(epsilons[i])", 
        legend = :bottomright,linestyle=linestyles[i],linecolor = linecolors[i] )
        scatter!(optbetas[i], evars[i], color = linecolors[i], label = "", markersize = 10,
                 markershape = :star)

    end
  
    xlabel!("β")
    ylabel!("h(β)")
    savefig(p,"evars.pdf")
 end

 # Plot for spectral radius vs. β; In TRC, erm values vs. β
 function  spectral_radius_unbounded_erm(betas,erm_trc,ϵ)

    beta1 = deepcopy(betas)
    max_beta = maximum(beta1)
    radiuses =[]
    for i in 1:7
        push!(beta1,max_beta+0.001*i)
    end

    #for β in betas
    for β in beta1
        spectral_radius= compute_rho(ϵ, β)
        push!(radiuses,spectral_radius)
    end

    p1 = plot(beta1,radiuses, linewidth=1, label = "spectral radius", 
             legend = :topleft,linestyle=:dot,
            #  xlims = (minimum(beta1),last(beta1)),
            #  ylims = (minimum(radiuses),last(radiuses)) 
             )
    xlabel!("β")
    ylabel!("Spectral radius")
    savefig(p1,"radius.pdf")

    p2 = plot(betas,erm_trc, linewidth=1, label = "ERM value",legend = :topright)
    xlabel!("β")
    ylabel!("ERM value")

    savefig(p2,"erm.pdf")
    
end

 function main()
    #pgfplotsx()
    
    evars = []
    # save spectral radius
    radiuses = []
    ϵ = 0.85
    δ = 0.01
    
    data = constrcut(ϵ)
    model = load_mdp(data)               
    initial_state_pro = [0.5,0.5,0]

    # #risk level of EVaR
    alpha_array = [0.7]

    # #Compute the optimal policy 
    evar, erm_trc, betas,opt_policy = compute_optimal_policy(alpha_array, initial_state_pro, model,δ)

    spectral_radius_unbounded_erm(betas,erm_trc,ϵ)

    evar_plot()
    

 

  

  
end 

########






main()