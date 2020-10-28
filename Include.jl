# Setup the paths -
const _PATH_TO_ROOT = pwd()

# need to use the package manager, to activate this project -
import Pkg
Pkg.activate(_PATH_TO_ROOT);

# external packages -
using Optim

# my codes -
include("Gibbs.jl")
include("Extent.jl")
