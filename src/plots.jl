using PlotlyJS, CSV, DataFrames

# Bar plot of the distribution of the final capital
function main()
  
    
    capitals = ["-1", "0", "1", "2", "3", "4",
          "5", "6", "7"]
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




main()