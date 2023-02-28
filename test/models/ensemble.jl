models = (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)

fparams = (
    AdaBoostClassifier=(:estimator, :estimators, :estimator_weights, :estimator_errors, :classes, :n_classes),
    BaggingClassifier=(:estimator, :base_estimator, :estimators, :estimators_samples, :estimators_features, :classes, :n_classes, :oob_score, :oob_decision_function),
    GradientBoostingClassifier=(:n_estimators, :feature_importances, :train_score, :loss, :init, :estimators, :oob_improvement),
    RandomForestClassifier=(:estimator, :estimators, :classes, :n_classes, :n_features, :n_outputs, :feature_importances, :oob_score, :oob_decision_function),
    ExtraTreesClassifier=(:estimator, :estimators, :classes, :n_classes, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_decision_function)
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
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

fparams = (
    AdaBoostRegressor=(:estimator, :estimators, :estimator_weights, :estimator_errors, :feature_importances),
    BaggingRegressor=(:estimator, :estimators, :estimators_samples, :estimators_features, :oob_score, :oob_prediction),
    GradientBoostingRegressor=(:feature_importances, :train_score, :loss, :init, :estimators, :oob_improvement),
    RandomForestRegressor=(:estimator, :estimators, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_prediction),
    ExtraTreesRegressor=(:estimators, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_prediction)
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
