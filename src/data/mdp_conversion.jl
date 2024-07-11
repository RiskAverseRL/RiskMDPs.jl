
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")

using CSV
using DataFrames, DataFramesMeta
using CSV: File


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

function  construct_transient_mpd(frame,states, actions, epsilon,filename)

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
            state_from = frame.idstatefrom[i]
            action = frame.idaction[i]
            state_to =  frame.idstateto[i]
            probability =  frame.probability[i] 
            reward =  frame.reward[i]
    
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
    filepath = joinpath(pwd(),"src","data",filename * "_tra.csv");
    data = DataFrame(idstatefrom = arr_idstatefrom, idaction = arr_idaction, idstateto = arr_idstateto, probability = arr_prob,
                  reward = arr_reward)
    CSV.write(filepath, data)
    end


function main()
        
        epsilon = 0.05
        filename::String = "ruin.csv"
        # .csv file is 1-based index
        filepath = joinpath(pwd(),"src","data",filename);
        file = DataFrame(File(filepath)); 
       
        states, actions = states_actions(file)
        #length(states))
        
        # Add a sink state and convert a discounted MDP to a transient MDP, epsilon is the 
        # probability of transienting to the sink state. epsilon = 1 - gamma
        construct_transient_mpd(file,states, actions,epsilon,filename)

    end

main()