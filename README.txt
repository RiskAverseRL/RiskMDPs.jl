8/8/2024
#-----------------------
erm_lp_test.jl

win_p represents the probability of winning one game.
π1,π3,and π7 are the optimal policies for mg0.8.csv.

when the winning probability is 0.75, that is mg0.75.csv,
 π1 ::Vector{Int} = [1,3,4,5,6,7,8,9,1,1]
 π3 ::Vector{Int} = [1,3,4,5,6,7,8,9,1,1]
 π7 ::Vector{Int} = [1,2,2,2,2,2,2,2,1,1]

when the winning probability is 0.85, that is mg0.85.csv,
 π1 ::Vector{Int} = [1,3,4,5,6,7,8,9,1,1]
 π3 ::Vector{Int} = [1,2,2,2,2,2,2,2,1,1]
 π7 ::Vector{Int} = [1,2,2,2,2,2,2,2,1,1]

#------------------------
modified_gamble.jl

Generate data file for the modified Gambler domain
mg0.75.csv, mg stands for modified gambler. 0.75 represents the probability of winning one game

The data files are saved \src\data

#---------------
ermLp.jl
Compute the optimal polices for EVaR and ERM

#-----------
Plots/figures
x.pdf files are saved in the folder RiskMDPs.jl