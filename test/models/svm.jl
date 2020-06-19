models = (
    SVMLinearClassifier,
    SVMClassifier,
    SVMNuClassifier,
)

fparams = (
    SVMLinearClassifier=(:coef, :intercept, :classes),
    SVMClassifier=(:support, :support_vectors, :n_support, :dual_coef, :coef,
                   :intercept, :fit_status, :classes),
    SVMNuClassifier=(:support, :support_vectors, :n_support, :dual_coef, :coef,
                     :intercept, :fit_status, :classes),
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

models = (
    SVMLinearRegressor,
    SVMRegressor,
    SVMNuRegressor,
)

fparams = (
    SVMLinearRegressor=(:coef, :intercept),
    SVMRegressor=(:support, :support_vectors, :dual_coef, :coef,
                  :intercept, :fit_status),
    SVMNuRegressor=(:support, :support_vectors, :dual_coef, :coef,
                    :intercept),
)

@testset "Fit/Predict" begin
    X, y, ls = simple_regression()
    for mod in models
        m = mod()
        r, f = test_regression(m, X, y, ls)
        @test r < 5
        @test keys(MB.fitted_params(m, f)) == getproperty(fparams, Symbol(mod))
    end
end
