using MDPs
using CSV
using Printf
using Plots

include(joinpath(@__DIR__, "VarDP3.jl"))
using .VarDP3

# -----------------------------
# Load MDP
# -----------------------------
mdp_path = joinpath(@__DIR__, "..", "data", "machine.csv")
mdp = VarDP3.load_intmdp(mdp_path)
S   = state_count(mdp)

# -----------------------------
# Parameters
# -----------------------------
T  = 10
α  = 0.1
t0 = 0
s0 = 1  # initial state for plots

rmax = VarDP3.max_abs_reward(mdp)

# -----------------------------
# Fixed τ-grid (same for all t)
# -----------------------------
grid_scale = 10.0   # widen grid to avoid clipping
Δτ         = 0.05    # grid step size

M = grid_scale * T * rmax
Xgrid = vcat(-Inf, collect(-M:Δτ:M), Inf)

# Same grid at every t
X = [Xgrid for _ in 0:T]

println("Using horizon T=$T, α=$α, grid_scale=$grid_scale, Δτ=$Δτ")

# -----------------------------
# Run DPs
# -----------------------------
println("\nRunning VaR-DP with side = :plus (h^+) ...")
V_plus,  π_plus  = VarDP3.vi(mdp, X, T; side=:plus)
println("  V_plus size: ", size(V_plus))   # length T+1 vector of S×K matrices
println("  π_plus size: ", size(π_plus))   # length T vector of S×K matrices

println("\nRunning VaR-DP with side = :minus (h^-) ...")
V_minus, π_minus = VarDP3.vi(mdp, X, T; side=:minus)
println("  V_minus size: ", size(V_minus))
println("  π_minus size: ", size(π_minus))

# -----------------------------
# VaR bounds at time t0 for all the states
# -----------------------------
VaR_minus_all = Vector{Union{Nothing,Float64}}(undef, S)  # VaR^- from V_minus
VaR_plus_all  = Vector{Union{Nothing,Float64}}(undef, S)  # VaR^+ from V_plus

for s in 1:S
    VaR_minus_all[s] = VarDP3.var(V_minus, X, t0, s, α)  # VaR^-
    VaR_plus_all[s]  = VarDP3.var(V_plus,  X, t0, s, α)  # VaR^+
end

println("\nVaR bounds at t=$t0 (left-quantile level α=$α)")
for s in 1:S
    println("State $s:  VaR^- = $(VaR_minus_all[s])   VaR^+ = $(VaR_plus_all[s])")
end

# -----------------------------
# Policy at VaR slices (time t0)
# Use hm/hp (h^- / h^+) for consistent grid snapping
# -----------------------------
println("\n" * repeat("=", 70))
println("POLICIES AT VaR SLICES (time t=$t0)")
println(repeat("=", 70))
println("State |   VaR^-    a^-   ||    VaR^+    a^+")
println(repeat("-", 55))

grid0 = X[t0+1]
mesh0 = VarDP3.Mesh(grid0)

for s in 1:S
    τm = VaR_minus_all[s]   # VaR^-
    τp = VaR_plus_all[s]    # VaR^+

    if τm === nothing || τp === nothing
        @printf("%5d |  (VaR not found on grid)\n", s)
        continue
    end

    km = VarDP3.hm(mesh0, τm)  # floor index (h^-)
    kp = VarDP3.hp(mesh0, τp)  # ceil index  (h^+)

    am = π_minus[t0+1][s, km]
    ap = π_plus[t0+1][s, kp]

    @printf("%5d | %8.2f %4d  || %8.2f %4d\n", s, τm, am, τp, ap)
end
println(repeat("-", 55))

# -----------------------------
# Save policy-at-VaR table to CSV
# -----------------------------
output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)

policy_file = joinpath(output_dir, "policy_at_VaR_t$(t0).csv")
open(policy_file, "w") do f
    write(f, "state,VaR_minus,action_minus,VaR_plus,action_plus\n")
    for s in 1:S
        τm = VaR_minus_all[s]
        τp = VaR_plus_all[s]
        if τm === nothing || τp === nothing
            continue
        end
        km = VarDP3.hm(mesh0, τm)
        kp = VarDP3.hp(mesh0, τp)
        am = π_minus[t0+1][s, km]
        ap = π_plus[t0+1][s, kp]
        write(f, "$s,$τm,$am,$τp,$ap\n")
    end
end
println("\n✓ Saved: $policy_file")

# -----------------------------
# Plots (state s0)
# -----------------------------
τgrid_all  = X[t0+1]
finite_idx = findall(isfinite, τgrid_all)
τgrid      = τgrid_all[finite_idx]

v_minus = vec(V_minus[t0+1][s0, finite_idx])
v_plus  = vec(V_plus[t0+1][s0,  finite_idx])

# Figure 1: value function bounds
p1 = plot(
    τgrid, v_minus,
    xlabel = "τ",
    ylabel = "v₀(s₀, τ)",
    label  = "v⁻ (h⁻)",
    lw     = 2,
    title  = "Lower/Upper Value Functions at t=$t0 from s₀=$s0"
)
plot!(p1, τgrid, v_plus, label="v⁺ (h⁺)", lw=2)
display(p1)

# Figure 2: VaR bounds for state s0 across multiple α values
alphas = collect(0.1:0.1:0.9)

p2 = plot(
    xlabel = "α",
    ylabel = "τ",
    xticks = alphas,
    title  = "VaR Bounds at t=$t0 for s₀=$s0",
    legend = :topleft
)

for αi in alphas
    VaR_minus_i = VarDP3.var(V_minus, X, t0, s0, αi)
    VaR_plus_i  = VarDP3.var(V_plus,  X, t0, s0, αi)

    if VaR_plus_i === nothing || VaR_minus_i === nothing
        annotate!(p2, αi, 0, text("VaR not found", 8))
        continue
    end

    plot!(p2, [αi, αi], [VaR_plus_i, VaR_minus_i], lw=6, label="α=$(αi)")
    scatter!(p2, [αi, αi], [VaR_plus_i, VaR_minus_i], ms=4, label="")
end

display(p2)
