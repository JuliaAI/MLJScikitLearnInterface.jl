using StableRNGs
using MLJScikitLearnInterface
using Test
import MLJBase
import ScikitLearn

const MB = MLJBase

include("testutils.jl")
include("macros.jl")

println("linear-regressors");       include("models/linear-regressors.jl")
println("linear-regressors-multi"); include("models/linear-regressors-multi.jl")

println("linear-classifiers");      include("models/linear-classifiers.jl")
println("discriminant-analysis");   include("models/discriminant-analysis.jl")
println("gaussian-process");        include("models/gaussian-process.jl")
println("ensemble");                include("models/ensemble.jl")
