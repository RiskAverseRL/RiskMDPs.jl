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

# Figures are saved in RiskMDPs folder
# plot evar vs α; β* vs. α
function plot_evar_alpha_beta(alpha_array)

    filepath00 = joinpath(pwd(),"src",  "data","0.0.csv");
    filepath05 = joinpath(pwd(),"src",  "data","0.05.csv");
    filepath10 = joinpath(pwd(),"src",  "data","0.1.csv");

    df00 = DataFrame(File(filepath00))
    df05 = DataFrame(File(filepath05))
    df10 = DataFrame(File(filepath10))

    evar00 = []
    beta00 = []
    evar05 = []
    beta05 = []
    evar10 = []
    beta10 = []

    for i in 1:size(df00,1)
       push!(evar00, df00.evar[i])
       push!(beta00,df00.beta[i])
       push!(evar05, df05.evar[i])
       push!(beta05,df05.beta[i])
       push!(evar10, df10.evar[i])
       push!(beta10,df10.beta[i])
    end
    
    #save_path = joinpath(pwd(),"src",  "data");
    p1=plot(alpha_array,evar00,label="penalty00", linewidth=2,seriestype=[:scatter :line])
    plot!(alpha_array,evar05,label="penalty05", linewidth=2,seriestype=[:scatter :line])
    plot!(alpha_array,evar10,label="penalty10", linewidth=2,seriestype=[:scatter :line])
    #xlims!(0.08,0.93)
   # ylims!(-10,-1)
    xlabel!("α")
    ylabel!("EVaR value function")
    savefig(p1,"evar_alpha.pdf")

    p2=scatter(alpha_array,beta00,label="penalty00", linewidth=2)
    #scatter!(alpha_array,beta05,label="penalty05", linewidth=2)
    scatter!(alpha_array,beta10,label="penalty10", linewidth=2)
    xlims!(0.08,0.93)
    xlabel!("α")
    ylabel!("β*")
    savefig(p2,"beta_alpha.pdf")
end


function main()
        # risk level of EVaR
        alpha_array = [0.1,0.2,0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
        # plot evar vs. α ; β* vs. α;
        plot_evar_alpha_beta(alpha_array)
end

main()