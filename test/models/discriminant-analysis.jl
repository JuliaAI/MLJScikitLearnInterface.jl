models = (
    BayesianLDA,
    BayesianQDA,
    )

fparams = (
    BayesianLDA=(:coef, :intercept, :covariance, :means, :priors, :scalings, :xbar, :classes, :explained_variance_ratio),
    BayesianQDA=(:covariance, :means, :priors, :rotations, :scalings),
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
