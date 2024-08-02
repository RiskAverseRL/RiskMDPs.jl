
using CSV
using DataFrames, DataFramesMeta
using CSV: File
using MDPs
using MDPs.Domains

# Get states and actions
function  states_actions(frame)

    state_space = [] # states are integer
    action_space = [] # actions are integer

    for i in 1:size(frame,1)

        if !(frame.idstateto[i] in state_space)
                push!(state_space,frame.idstateto[i] )
        end

        if !(frame.idstatefrom[i] in state_space)
                push!(state_space,frame.idstatefrom[i] )
        end
        if !(frame.idaction[i] in action_space)
                push!(action_space,frame.idaction[i] )
        end

    state_space = sort(state_space)
    action_space = sort(action_space)
    
  end
    return state_space,action_space
end

# convert 0-based discounted mdp to 1-based transient mdp
#function  construct_transient_mpd(frame,states, actions, epsilon,filename)
function  construct_transient_mpd(frame,states, actions, epsilon)
        state_size = length(states)
        action_size = length(actions)
        
        # reward of going from state s to state s’ through action a.
        r = zeros((state_size+1, action_size,state_size+1))
        # transition probablity of (state, action, next state, model)
        p = zeros((state_size+1, action_size,state_size+1))
    
        e = state_size+1 # sink state

        arr_idstatefrom = Vector{Int}()
        arr_idstateto = Vector{Int}()
        arr_idaction = Vector{Int}()
        arr_prob = Vector{Float64}()
        arr_reward = Vector{Float64}()

        for i in 1:size(frame,1)
            # convert 0_based to 1_based for states and actions
            state_from = frame.idstatefrom[i]+1
            action = frame.idaction[i]+1
            state_to =  frame.idstateto[i]+1
            probability =  frame.probability[i] 
            reward =  frame.reward[i]
            
            # Transient to a non-sink state
            p[state_from, action, state_to] = probability * (1- epsilon)
            r[state_from, action, state_to] = reward

            push!(arr_idstatefrom,state_from)
            push!(arr_idaction, action)
            push!(arr_idstateto, state_to)
            push!(arr_prob,p[state_from, action, state_to])
            push!(arr_reward, r[state_from, action, state_to])

            # Transient to the sink state
            p[state_from, action, e] = probability * epsilon
            r[state_from, action, e] = 0
            push!(arr_idstatefrom,state_from)
            push!(arr_idaction, action)
            push!(arr_idstateto, e)
            push!(arr_prob,p[state_from, action, e])
            push!(arr_reward, r[state_from, action,e])
    
        end      
        
        # Add the sink state transients to the sink state, 
        # Assume there is only one action available
        p[e, 1, e] = 1
        r[e, 1, e] = 0
        push!(arr_idstatefrom,e)
        push!(arr_idaction, 1)
        push!(arr_idstateto, e)
        push!(arr_prob,p[e, 1, e])
        push!(arr_reward, r[e, 1, e] )

    # save the transient MDP
    data = DataFrame(idstatefrom = arr_idstatefrom, idaction = arr_idaction, idstateto = arr_idstateto, probability = arr_prob,
                  reward = arr_reward)
    data
    end

# Sum the probabilities with the same s,a,s' and r
function compress_transient_mpd(frame,states, actions,capital)
    state_size = length(states)
    action_size = length(actions)
    
    # reward of going from state s to state s’ through action a.
    r = zeros((state_size+1, action_size,state_size+1))
    # transition probablity of (state, action, next state, model)
    p = zeros((state_size+1, action_size,state_size+1))

    e = state_size+1 # sink state

    arr_idstatefrom = Vector{Int}()
    arr_idstateto = Vector{Int}()
    arr_idaction = Vector{Int}()
    arr_prob = Vector{Float64}()
    arr_reward = Vector{Float64}()

    c = Dict{Tuple{Int64, Int64,Int64,Float64},  Any}()
    for i in 1:size(frame,1)
        state_from = frame.idstatefrom[i]
        action = frame.idaction[i]
        state_to =  frame.idstateto[i]
        probability =  frame.probability[i] 
        reward =  frame.reward[i]

        if haskey(c,(state_from,action,state_to,reward)) == false
            c[(state_from,action,state_to,reward)] = probability
        else
            c[(state_from,action,state_to,reward)] += probability 
        end
    end
    for (key, value ) in c
        push!(arr_idstatefrom,key[1])
        push!(arr_idaction, key[2])
        push!(arr_idstateto, key[3])
        push!(arr_prob,value)
        push!(arr_reward, key[4])
    end
       # save the compressed transient MDP
       filepath = joinpath(pwd(),"src",  "data","g"*"$capital"* ".csv");
       data = DataFrame(idstatefrom = arr_idstatefrom, idaction = arr_idaction, idstateto = arr_idstateto, probability = arr_prob,
                     reward = arr_reward)
       data_sorted = sort(data,[:idstatefrom,:idaction,:idstateto])
       CSV.write(filepath, data_sorted)
end

function main()
        #epsilon is the probability of transienting to the sink state. 
        # epsilon = 1 - gamma, gamma is the discount factor in a discounted MDP
        epsilon = 0.1
        
        # p: the probability of winning one game
        p =0.8
        initial_capital = 5

        # Generate a 0_based dataframe from Gambler domain, 
        model = Gambler.Ruin(p, initial_capital)
        file = MDPs.save_mdp(DataFrame, model) # DataFrame

        states, actions = states_actions(file)
        
        
        # Add a sink state to the 0_based MDP and convert it to a 1_based transient MDP, 
        #data_transient = construct_transient_mpd(file,states, actions,epsilon,filename)
        data_transient = construct_transient_mpd(file,states, actions,epsilon)

        #compress the transition probabilities with same s,a,s',r)
        compress_transient_mpd(data_transient,states, actions,initial_capital)

    end

main()