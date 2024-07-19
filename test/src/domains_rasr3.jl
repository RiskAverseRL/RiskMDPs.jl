using RobustRL, RobustRL.Domains

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
    # inventory
    begin
        # risk parameters
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.8
        δ = 5.            # discretization error
        ΔR = 5. / (1-γ)
        initstate = 1         # initial state
        horizon = 100
        # Define the inventory model
        demand = Inventory.Demand([0,2,3,4,5,30,3,2],
                                [0.1,0.3,0.1,0.1,0.1,0.1,0.0,0.2])
        costs = Inventory.Costs(5.,2.,0.3,0.5)
        limits = Inventory.Limits(100, 0, 50)
        params = Inventory.Parameters(demand, costs, 16., limits)
        model = Inventory.Model(params)
        problems["inventory"] =  Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    #invetory_generic
    begin
        model = load_generic_mdp(File("inventory.csv"))
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.9
        δ = 1.
        ΔR = 1. / (1-γ)
        initstate = 1         # initial state
        horizon = 100
        problems["inventory_generic"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    # machine
    begin
        # risk parameters
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.8
        δ = 2.            # discretization error
        ΔR = 20. / (1-γ)
        initstate = 1         # initial state
        horizon = 100
        model = Machine.Replacement()
        problems["machine"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
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
    # riverswim
    begin
        model = load_generic_mdp(File("riverswim.csv"))
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.98
        horizon = 100
        initstate = 1         # initial state
        δ = 1.          # discretization error
        ΔR = 1 / (1-γ)
        problems["riverswim"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    # cancer
    begin
        model = load_generic_mdp(File("cancer.csv"))
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.9
        horizon = 6
        initstate = 1         # initial state
        δ = 0.1          # discretization error
        ΔR = 0.1 / (1-γ)
        problems["cancer"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    # population
    begin
        model = load_generic_mdp(File("population.csv"))
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.7
        horizon = 50
        initstate = 1         # initial state
        δ = 100.          # discretization error
        ΔR = 100. / (1-γ)
        problems["cancer"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    # onestatepm
    begin
        model = Simple.OneStatePlusMinus(100)
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.95
        initstate = 1         # initial state
        horizon = 100
        δ = 1.          # discretization error
        ΔR = 1 / (1-γ)
        problems["onestatepm"] = Problem(α, β, γ, δ, ΔR, horizon, initstate, model)
    end
    problems
end    
