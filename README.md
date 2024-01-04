# RiskMDPs

Solving MDPs with risk measure objectives

## Installation

The package is not yet avalaible through the julia repository. You need to install it directly from github.

### Using the package

The simplest way to install this package is as follows.  
```julia
using Pkg
Pkg.add(url="https://github.com/RiskAverseRL/MDPs.jl.git")
Pkg.add(url="https://github.com/RiskAverseRL/RiskMDPs.jl.git")
```

### Developing the package

This installation may have some problems with running tests. The more convoluted way, but one that should work is to do the following:

1. Clone the repositories

```bash
git clone git@github.com:RiskAverseRL/MDPs.jl.git
git clone git@github.com:RiskAverseRL/RiskMDPs.jl.git
```

2. Register the packages for development (note that you can start the package mode by typing ])

```julia
pkg> develop MDPs.jl
pkg> develop RiskMDPs.jl
```

3. Create a local registry by following the instructions at: 

<https://github.com/GunnarFarneback/LocalRegistry.jl>

3. Register the MDP package

```julia
julia> using LocalRegistry
julia> register("MDPs")
```

Remember that you may need to call the register function again when there is any change to the MDP package.

4. You should be all set now. You can test your installation by running

```julia
pkg> test MDPs
pkg> test RiskMDPs
```
