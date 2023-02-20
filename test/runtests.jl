using StableRNGs
using MLJScikitLearnInterface
using Test
import MLJBase
import ScikitLearn
import MLJBase: target_scitype, input_scitype, output_scitype

const MB = MLJBase

include("testutils.jl")
include("macros.jl")

println("\nlinear-regressors");       include("models/linear-regressors.jl")
println("\nlinear-regressors-multi"); include("models/linear-regressors-multi.jl")

println("\nlinear-classifiers");      include("models/linear-classifiers.jl")
println("\ndiscriminant-analysis");   include("models/discriminant-analysis.jl")
println("\ngaussian-process");        include("models/gaussian-process.jl")
println("\nensemble");                include("models/ensemble.jl")
println("\nclustering");              include("models/clustering.jl")

prinln("\ngeneric interface tests");  include("generic_api_tests.jl")
