models = (
    MultiTaskLassoRegressor,
    MultiTaskLassoCVRegressor,
    MultiTaskElasticNetRegressor,
    MultiTaskElasticNetCVRegressor
    )

fparams = (
    MultiTaskLassoRegressor=(:coef, :intercept),
    MultiTaskLassoCVRegressor=(:coef, :intercept, :alpha, :mse_path, :alphas),
    MultiTaskElasticNetRegressor=(:coef, :intercept),
    MultiTaskElasticNetCVRegressor=(:coef, :intercept, :alpha, :mse_path, :l1_ratio)
    )


@testset "Fit/Predict" begin
    # check roughly that models return sensible results
    X, y, ls = simple_regression_2()
    for mod in models
        m = mod()
        # ratio of rmse to LS
        r, f = test_regression_2(m, X, y, ls)
        @test r < 200
        @test keys(MB.fitted_params(m, f)) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end
