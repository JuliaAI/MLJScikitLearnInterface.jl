models = (
    ARDRegressor,
    BayesianRidgeRegressor,
    ElasticNetRegressor,
    ElasticNetCVRegressor,
    HuberRegressor,
    LarsRegressor,
    LarsCVRegressor,
    LassoRegressor,
    LassoCVRegressor,
    LassoLarsRegressor,
    LassoLarsCVRegressor,
    LassoLarsICRegressor,
    LinearRegressor,
    OrthogonalMatchingPursuitRegressor,
    OrthogonalMatchingPursuitCVRegressor,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    RidgeRegressor,
    RidgeCVRegressor,
    SGDRegressor,
    TheilSenRegressor
    )

fparams = (
    ARDRegressor=(:coef, :intercept, :alpha, :lambda, :sigma, :scores),
    BayesianRidgeRegressor=(:coef, :intercept, :alpha, :lambda, :sigma, :scores),
    ElasticNetRegressor=(:coef, :intercept),
    ElasticNetCVRegressor=(:coef, :intercept, :l1_ratio, :mse_path, :alphas),
    HuberRegressor=(:coef, :intercept, :scale, :outliers),
    LarsRegressor=(:coef, :intercept, :alphas, :active, :coef_path),
    LarsCVRegressor=(:coef, :intercept, :alpha, :alphas, :cv_alphas, :mse_path, :coef_path),
    LassoRegressor=(:coef, :intercept),
    LassoCVRegressor=(:coef, :intercept, :alpha, :alphas, :mse_path, :dual_gap),
    LassoLarsRegressor=(:coef, :intercept, :alphas, :active, :coef_path),
    LassoLarsCVRegressor=(:coef, :intercept, :coef_path, :alpha, :alphas, :cv_alphas, :mse_path),
    LassoLarsICRegressor=(:coef, :intercept, :alpha),
    LinearRegressor=(:coef, :intercept),
    OrthogonalMatchingPursuitRegressor=(:coef, :intercept),
    OrthogonalMatchingPursuitCVRegressor=(:coef, :intercept, :n_nonzero_coefs),
    PassiveAggressiveRegressor=(:coef, :intercept),
    RANSACRegressor=(:estimator, :n_trials, :inlier_mask, :n_skips_no_inliers, :n_skips_invalid_data, :n_skips_invalid_model),
    RidgeRegressor=(:coef, :intercept),
    RidgeCVRegressor=(:coef, :intercept, :alpha, :cv_values),
    SGDRegressor=(:coef, :intercept, :average_coef, :average_intercept),
    TheilSenRegressor=(:coef, :intercept, :breakdown, :n_subpopulation)
    )

@testset "Fit/Predict" begin
    # check roughly that models return sensible results
    X, y, ls = simple_regression()
    for mod in models
        if mod in (ElasticNetRegressor, LassoRegressor, LassoLarsRegressor,
                   OrthogonalMatchingPursuitRegressor, OrthogonalMatchingPursuitCVRegressor
                   )
            continue
        end
        m = mod()
        # ratio of rmse to LS
        r, f = test_regression(m, X, y, ls)
        @test r < 3 # no more than twice as bad
        @test keys(MB.fitted_params(m, f)) == getproperty(fparams, Symbol(mod))
    end
    # these models are pretty bad because their parameters are not set
    for mod in (ElasticNetRegressor, LassoRegressor, LassoLarsRegressor)
        m = mod()
        r, f = test_regression(m, X, y, ls)
        # these models are pretty bad because their parameters are bad
        @test r < 15
        @test keys(MB.fitted_params(m, f)) == getproperty(fparams, Symbol(mod))
    end
    # Both of those are a bit specific (more aimed at finding sparsity mask)
    # and if used "just like that" will show unsightly error messages
    # OrthogonalMatchingPursuitCVRegressor
    # OrthogonalMatchingPursuitRegressor
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end
