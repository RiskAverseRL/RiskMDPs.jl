#using PlotlyJS, CSV, DataFrames
using Plots

# Bar plot of the distribution of the final capital
function main()
    capitals = ["-1", "0", "1", "2", "3", "4", "5", "6", "7"]
    # The distribution is from "data_bar.csv"
    α9 = [0.19042857142857142,0,0,0,0,0,0,0,0.8095714285714286]
    α7 = [0.11985714285714286,0,0,0,0,0,0,0,0.8801428571428571]
    α4 = [0,0,0.2591428571428571,0,0,0,0,0,0.7408571428571429]
    α2 = [0,0,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,
          0.14285714285714285,0.14285714285714285,0.14285714285714285]

p = plot([
     bar(x=capitals, y=α9, name="α = 0.9"),
     bar(x=capitals, y=α7, name="α = 0.7"),
     bar(x=capitals, y=α4, name="α = 0.4"),
     bar(x=capitals, y=α2, name="α = 0.2")
], Layout(barmode="group",xticks = -1:1:7, yticks = 0:0.1:1, yaxis_title_text="relative frequency",
       xaxis_title_text="final capital",template=:plotly_white,xaxis = attr(
        tickmode = "array",
        tickvals = [-1,0,1,2,3,4,5,6,7],
        ticktext = ["-1", "0", "1", "2", "3", "4", "5","6"]
    ),legend=:innerbottomright))

savefig(p,"final_capital_distri.pdf")

end




#main()
#####

function main2()
    pgfplotsx()
    p1 = scatter(capitals, α9, label = "α = 0.9", size=(350,240), legend=:topleft,
                 xlabel="Capital", ylabel = "Probability")
    scatter!(capitals, α7, label = "α = 0.7")
    scatter!(capitals, α4, label = "α = 0.4")
    scatter!(capitals, α2,  label = "α = 0.2")
    plot!(capitals, α9, linestyle=:dash, linecolor=p1.series_list[1][:fillcolor], label=nothing)
    plot!(capitals, α7, linestyle=:dash, linecolor=p1.series_list[2][:fillcolor], label=nothing)
    plot!(capitals, α4, linestyle=:dash, linecolor=p1.series_list[3][:fillcolor], label=nothing)
    plot!(capitals, α2, linestyle=:dash, linecolor=p1.series_list[4][:fillcolor], label=nothing)
    p1
end

p1 = main2()

savefig(p1,"final_capital_distri.pdf")

