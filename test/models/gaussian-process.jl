models = (
    GaussianProcessRegressor,
    GaussianProcessClassifier,
    )

fparams = (
    GaussianProcessRegressor=(:X_train, :y_train, :kernel, :L, :alpha, :log_marginal_likelihood_value),
    GaussianProcessClassifier=(:kernel, :log_marginal_likelihood_value, :classes, :n_classes),
    )

@testset "Fit/Predict" begin
    X, y = simple_binaryclf()
    m = GaussianProcessClassifier()
    acc, f = test_clf(m, X, y)
    @test acc >= 0.75
    @test keys(MB.fitted_params(m, f)) == fparams.GaussianProcessClassifier

    X, y, ls = simple_regression()
    m = GaussianProcessRegressor()
    r, f = test_regression(m, X, y, ls)
    @test r < 0.1 # super over-fitting
end

@testset "Docstrings" begin
    for mod in models
        for mod in models
            m = mod()
            @test !isempty(MB.docstring(m))
        end
    end
end
