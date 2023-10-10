
clf_models = (
    # discriminant analysis
    BayesianLDA,
    
    # ensemble classifiers
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,

    # linear classifiers
    LogisticClassifier,
    LogisticCVClassifier,
    PassiveAggressiveClassifier,
    PerceptronClassifier,
    RidgeClassifier,
    RidgeCVClassifier,
    SGDClassifier,

    # svc
    SVMLinearClassifier
)

@testset "Classification Feature Importance" begin
    X, y = simple_binaryclf()
    num_columns = length(Tables.columnnames(X))
    for mod in clf_models
        m = mod()
        f, _, r = MB.fit(m, 1, X, y)
        fi = MB.feature_importances(m, f, r)
        @test size(fi) == (num_columns,)
    end
end

reg_models = (
    # ensemble regressors
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,

    # linear regressors
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
    RidgeRegressor,
    RidgeCVRegressor,
    SGDRegressor,
    TheilSenRegressor,

    # srv
    SVMLinearRegressor
)

@testset "Regression Feature Importance" begin
    X, y = MB.make_regression()
    num_columns = length(Tables.columnnames(X))
    for mod in reg_models
        m = mod()
        f, _, r = MB.fit(m, 1, X, y)
        fi = MB.feature_importances(m, f, r)
        @test size(fi) == (num_columns,)
    end
end

multi_reg_models = (
    MultiTaskLassoRegressor,
    MultiTaskLassoCVRegressor,
    MultiTaskElasticNetRegressor,
    MultiTaskElasticNetCVRegressor
)

@testset "Multi-Task Regression Feature Importance" begin
    X, y, _ = simple_regression_2()
    num_columns = size(X, 2)
    for mod in multi_reg_models
        m = mod()
        f, _, r = MB.fit(m, 1, X, y)
        fi = MB.feature_importances(m, f, r)
        @test size(fi) == (num_columns,)
    end
end
