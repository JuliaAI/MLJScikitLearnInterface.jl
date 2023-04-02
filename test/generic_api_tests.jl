using MLJTestInterface
import MLJBase
import MLJBase: finaltypes,
    Continuous,
    Finite,
    Multiclass,
    target_scitype,
    input_scitype,
    is_supervised,
    package_name,
    Model,
    Table

const MODELS = filter(finaltypes(Model)) do m
    package_name(m) == "MLJScikitLearnInterface"
end

const SINGLE_TARGET_REGRESSORS = filter(MODELS) do m
     AbstractVector{Continuous} <: target_scitype(m) &&
        is_supervised(m)
end

const SINGLE_TARGET_CLASSIFIERS = filter(MODELS) do m
    AbstractVector{Multiclass{3}} <: target_scitype(m) &&
        AbstractVector{Multiclass{2}} <: target_scitype(m) &&
        Table(Continuous) <: input_scitype(m) &&
        is_supervised(m)
end

const TRANSFORMERS = filter(MODELS) do m
     Table(Continuous) <: input_scitype(m) &&
         !is_supervised(m) &&
         # excluded because it does not converge:
         m != AffinityPropagation
end

bad_single_target_classifiers = [
    # https://github.com/JuliaAI/MLJScikitLearnInterface.jl/issues/48
    PassiveAggressiveClassifier,
    PerceptronClassifier,
    SGDClassifier,
    SVMClassifier,
    SVMNuClassifier,
    BayesianQDA,
    ProbabilisticSGDClassifier,
    SVMLinearClassifier,
    SVMLinearClassifier,
    LogisticCVClassifier,
    LogisticClassifier,
    GaussianProcessClassifier,
    GradientBoostingClassifier
]

@test_broken isempty(bad_single_target_classifiers)

@testset "generic interface tests" begin
    @testset "regressors"  begin
        failures, summary = MLJTestInterface.test(
            SINGLE_TARGET_REGRESSORS,
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "classifiers" begin
        for data in [
            MLJTestInterface.make_binary(),
            MLJTestInterface.make_multiclass(),
        ]
            failures, summary = MLJTestInterface.test(
                setdiff(
                    SINGLE_TARGET_CLASSIFIERS,
                    bad_single_target_classifiers,
                ),
                data...;
                mod=@__MODULE__,
                verbosity=0, # bump to debug
                throw=false  # set to true to debug
            )
            @test isempty(failures)
        end
    end
    @testset "transformers" begin
        failures, summary = MLJTestInterface.test(
            TRANSFORMERS,
            MLJTestInterface.make_regression()[1];
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
end

true
