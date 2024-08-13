8/13/2024
#-----------------------
erm_lp_test.jl

"7 mgp0.68.csv" is generated from "modified_gamble.jl".

1) Simulate ERM value functions;
2) save the distribution of final capital to
"data_bar.csv" under \src\data.

#------------------------
modified_gamble.jl

Gambler(win_p,capital)

It needs needs parameters capital and win_p. 
The data file is named "$capital mgp$win_p.csv" and saved \src\data.

#---------------
ermLp.jl
1) Compute the optimal polices for EVaR
2)  Plot for unbounded ERM values in TRC and constant erm values in a discounted criterion
3) plot the optimal policies
4) The plot is saved .\RiskMDP.jl
#-----------
Plots
1) plot the distribution of final capital
2) The plot is saved .\RiskMDP.jl