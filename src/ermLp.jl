using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using MDPs
using JuMP, HiGHS
using CSV: File
using RiskMDPs
using Plots
using Infiltrator
using CSV

# ---------------------------------------------------------------
# ERM with the total reward criterion. an infinite horizon. This formulation is roughly equivalent 
# to using a discount factor of 1.0. The last state is considered as the sink state
# 1) compute the optimal policy for EVaR and ERM
# 2) Plot for unbounded ERM values
# 3) save data of evars, optimal beta values, alphas to csv files
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
    
     lpm = Model(HiGHS.Optimizer)
     set_silent(lpm)

     state_number = state_count(model)
     π = zeros(Int , state_number)


    @variable(lpm, w[1:(state_number-1)] .<= -1e-10)
    @objective(lpm, Min, sum(w))

    constraints::Vector{Vector{ConstraintRef}} = []

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

    # The constraint for the sink state
    #push!(constraints, [@constraint(lpm, w[state_number] == -1)])
    optimize!(lpm)

    # Check if the linear program has a feasible solution
    # value.(w)[(state_number -1 )] =0.0 is a feasible solution for linear program, but it is 
    # unbounded for regular erm value function
    if termination_status(lpm) ==  DUAL_INFEASIBLE || value.(w)[(state_number -1 )] == 0.0
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


# Compute the optimal policy for different α values
function compute_optimal_policy(alpha_array,initial_state_pro, model,δ, ΔR)
    
    #Save erm values and beta values for plotting unbounded erm
    # in transient mdp 
    erm_values = []
    beta_values =[]

    # Evar values and the optimal β values
    evars = []
    betaopt = []

    for α in alpha_array
        βs =  evar_discretize_beta(α, δ, ΔR)
        max_h =-Inf
        optimal_policy = []
        optimal_beta = -1
        optimal_v = []
        optimal_w = []


        for β in βs
            B = compute_B(model,β)
            status,w,v,π = erm_linear_program(model,B,β)
            
            # compute the optimal policy for β that has only feasible solution
            if cmp(status,"infeasible") ==0 
                #break
                continue
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
        println(" vector of exponential erm value is  ",optimal_w)
        push!(evars, max_h)
        push!(betaopt,optimal_beta)
    end
    (erm_values,beta_values, evars,betaopt)
end

# Compute a single ERM value using the vector of regular value function and initial distribution
function compute_erm(value_function :: Vector, initial_state_pro :: Vector)
    return sum(value_function.*initial_state_pro)
end

 # save alpha values and evar values to csv files 
function save_reward_alpha_evar(alpha_array,evars,betaopt,penaltyr)

    filepath = joinpath(pwd(),"src",  "data","$penaltyr"* ".csv");
    data = DataFrame(alpha = alpha_array, evar = evars,beta=betaopt)
    data_sorted = sort(data,[:alpha])
    CSV.write(filepath, data_sorted)

end

# Show unbounded ERM value functions
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

function main_evar()

    δ = 0.005
    ΔR =1

    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                   "data", "gm0.1.csv")
    # filepath = joinpath(dirname(pathof(RiskMDPs)), 
    #                    "data", "single_tra_dis.csv")
                                 
    model = load_mdp(File(filepath))
    
    # Uniform initial state distribution
    state_number = state_count(model)
    initial_state_pro = Vector{Float64}()
    for index in 1:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-1)) # start with a non-sink state
    end
    append!(initial_state_pro,0) # add the sink state with the initial probability 0

    # risk level of EVaR
    alpha_array = [0.1,0.2,0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]

    #Compute the optimal policy 
    erm_trc, betas,evars,betaopt = compute_optimal_policy(alpha_array,initial_state_pro, model,δ, ΔR)

    # The variable rpunish means the penalty(-penaltyr) for keep playing
    # Save α, evar, and β*; file name is gm$penaltyr.csv
    penaltyr = 0.1
    save_reward_alpha_evar(alpha_array,evars,betaopt,penaltyr)

    #Unbounded ERM value functions
    # plot erm values in a discounted MDP and a transient MDP
    # erm_discounted = r/(1-γ)
    erm_discounted = -2
    #erms_dis_trc(erm_trc, betas, erm_discounted)

    
    # Value iteration
    # v,_,_ = value_iteration(model, InfiniteH(0.7); iterations = 10000)
    # println("---------------------------")
    # println("expectation, value functions: ",v)
  
end 

########

function main_erm()

    filepath = joinpath(dirname(pathof(RiskMDPs)), 
                   "data", "gm0.1.csv")
    model = load_mdp(File(filepath))
    
    state_number = state_count(model)
    initial_state_pro = Vector{Float64}()
    for index in 1:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-1)) # start with a non-sink state
    end
    append!(initial_state_pro,0) # add the sink state with the initial probability 0

    
    β_array = [ 0.000001,0.01,0.1,10,20]

    for β ∈ β_array
        B = compute_B(model, β)
        status,w,v,π = erm_linear_program(model, B, β)
        println("++++++++++++++++++++++++++++++++++  ", β)
        #println("B = \n $")
        #println("-----------------")
        #println("w =\n", w)
        #println("-----------------")
       # println("v =\n", v)
        println("-----------------")
        println("π =\n", π)
    end
end 

####

main_evar()







