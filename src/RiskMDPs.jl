module RiskMDPs

using MDPs
using RiskMeasures

include("nested.jl")
export NestedFiniteH, NestedInfiniteH
export horizon, discount

include("erm.jl")
export DiscountedERM, IndefiniteERM
export mrp_exp, mrp_exp!

include("evar.jl")
export DiscountedEVaR

include("utility.jl")
export AugmentedMarkov
export AugUtility, UtilityVaR, UtilityCVaR, UtilityEVaR

end # module RiskMDPs
