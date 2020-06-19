models = (
    LogisticClassifier,
    LogisticCVClassifier,
    PassiveAggressiveClassifier,
    PerceptronClassifier,
    RidgeClassifier,
    RidgeCVClassifier,
    SGDClassifier,
    ProbabilisticSGDClassifier,
    )

fparams = (
    LogisticClassifier=(:classes, :coef, :intercept),
    LogisticCVClassifier=(:classes, :coef, :intercept, :Cs, :l1_ratios, :coefs_paths, :scores, :C, :l1_ratio),
    PassiveAggressiveClassifier=(:coef, :intercept),
    PerceptronClassifier=(:coef, :intercept),
    RidgeClassifier=(:coef, :intercept),
    RidgeCVClassifier=(:coef, :intercept),
    SGDClassifier=(:coef, :intercept),
    ProbabilisticSGDClassifier=(:coef, :intercept)
    )

@testset "Fit/Predict" begin
    X, y = simple_binaryclf()
    for mod in models
        m = mod()
        acc, f = test_clf(m, X, y)
        @test acc >= 0.75
        @test keys(MB.fitted_params(m, f)) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end
