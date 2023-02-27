const AdaBoostRegressor_ = sken(:AdaBoostRegressor)
@sk_reg mutable struct AdaBoostRegressor <: MMI.Deterministic
    estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    loss::String           = "linear"::(_ in ("linear","square","exponential"))
    random_state::Any      = nothing
end
MMI.fitted_params(model::AdaBoostRegressor, (f, _, _)) = (
    estimators           = f.estimators_,
    estimator_weights    = f.estimator_weights_,
    estimator_errors     = f.estimator_errors_,
    feature_importances  = f.feature_importances_
    )
add_human_name_trait(AdaBoostRegressor, "AdaBoost ensemble regression")

# ----------------------------------------------------------------------------
const AdaBoostClassifier_ = sken(:AdaBoostClassifier)
@sk_clf mutable struct AdaBoostClassifier <: MMI.Probabilistic
    estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    algorithm::String      = "SAMME.R"::(_ in ("SAMME", "SAMME.R"))
    random_state::Any      = nothing
end
MMI.fitted_params(m::AdaBoostClassifier, (f, _, _)) = (
    estimators        = f.estimators_,
    estimator_weights = f.estimator_weights_,
    estimator_errors  = f.estimator_errors_,
    classes           = f.classes_,
    n_classes         = f.n_classes_
    )
meta(AdaBoostClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    )

# ============================================================================
const BaggingRegressor_ = sken(:BaggingRegressor)
@sk_reg mutable struct BaggingRegressor <: MMI.Deterministic
    estimator::Any      = nothing
    n_estimators::Int        = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool          = true
    bootstrap_features::Bool = false
    oob_score::Bool          = false
    warm_start::Bool         = false
    n_jobs::Option{Int}      = nothing
    random_state::Any        = nothing
    verbose::Int             = 0
end
MMI.fitted_params(model::BaggingRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    estimators_samples  = f.estimators_samples_,
    estimators_features = f.estimators_features_,
    oob_score           = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = model.oob_score ? f.oob_prediction_ : nothing
    )
add_human_name_trait(BaggingRegressor, "bagging ensemble regressor")

# ----------------------------------------------------------------------------
const BaggingClassifier_ = sken(:BaggingClassifier)
@sk_clf mutable struct BaggingClassifier <: MMI.Probabilistic
    estimator::Any      = nothing
    n_estimators::Int        = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool          = true
    bootstrap_features::Bool = false
    oob_score::Bool          = false
    warm_start::Bool         = false
    n_jobs::Option{Int}      = nothing
    random_state::Any        = nothing
    verbose::Int             = 0
end
MMI.fitted_params(m::BaggingClassifier, (f, _, _)) = (
    base_estimator        = f.base_estimator_,
    estimators            = f.estimators_,
    estimators_samples    = f.estimators_samples_,
    estimators_features   = f.estimators_features_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
meta(BaggingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "bagging ensemble classifier"
    )

# ============================================================================
const GradientBoostingRegressor_ = sken(:GradientBoostingRegressor)
@sk_reg mutable struct GradientBoostingRegressor <: MMI.Deterministic
    loss::String                    = "squared_error"::(_ in ("squared_error","absolute_error","huber","quantile"))
    learning_rate::Float64          = 0.1::(_>0)
    n_estimators::Int               = 100::(_>0)
    subsample::Float64              = 1.0::(_>0)
    criterion::String               = "friedman_mse"::(_ in ("squared_error","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int                 = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
    init::Any                      = nothing
    random_state::Any              = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    alpha::Float64                 = 0.9::(_>0)
    verbose::Int                   = 0
    max_leaf_nodes::Option{Int}    = nothing::(_===nothing || _>0)
    warm_start::Bool               = false
#    presort::Union{Bool,String}    = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64   = 0.1::(_>0)
    n_iter_no_change::Option{Int}  = nothing
    tol::Float64                   = 1e-4::(_>0)
end
MMI.fitted_params(m::GradientBoostingRegressor, (f, _, _)) = (
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )
add_human_name_trait(GradientBoostingRegressor, "gradient boosting ensemble regression")

# ----------------------------------------------------------------------------
const GradientBoostingClassifier_ = sken(:GradientBoostingClassifier)
@sk_clf mutable struct GradientBoostingClassifier <: MMI.Probabilistic
    # TODO: Remove "deviance" when python sklearn releases v1.3.0
    loss::String                    = "log_loss"::(_ in ("deviance", "log_loss","exponential"))
    learning_rate::Float64          = 0.1::(_>0)
    n_estimators::Int               = 100::(_>0)
    subsample::Float64              = 1.0::(_>0)
    criterion::String               = "friedman_mse"::(_ in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int                 = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
    init::Any                      = nothing
    random_state::Any              = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    verbose::Int                   = 0
    max_leaf_nodes::Option{Int}    = nothing::(_===nothing || _>0)
    warm_start::Bool               = false
#    presort::Union{Bool,String}    = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64   = 0.1::(_>0)
    n_iter_no_change::Option{Int}  = nothing
    tol::Float64                   = 1e-4::(_>0)
end
MMI.fitted_params(m::GradientBoostingClassifier, (f, _, _)) = (
    n_estimators        = f.n_estimators_,
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    ## TODO: Remove the `loss_` attribute when python sklearn releases v1.3
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )
meta(GradientBoostingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )

# ============================================================================
const RandomForestRegressor_ = sken(:RandomForestRegressor)
@sk_reg mutable struct RandomForestRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "squared_error"::(_ in ("squared_error","absolute_error", "friedman_mse", "poisson"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    ## TODO: Remove the "auto" option in python sklearn v1.3
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
    ccp_alpha::Float64             =0.0::(_ ≥ 0)
    max_samples::Union{Nothing,Float64,Int} =
        nothing::(_ === nothing || (_ ≥ 0 && (_ isa Integer || _ ≤ 1)))
end
MMI.fitted_params(model::RandomForestRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    feature_importances = f.feature_importances_,
    n_features          = f.n_features_in_,
    n_outputs           = f.n_outputs_,
    oob_score           = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = model.oob_score ? f.oob_prediction_ : nothing
    )
meta(RandomForestRegressor,
    input   = Table(Count,Continuous),
    target  = AbstractVector{Continuous},
    weights = false
    )

# ----------------------------------------------------------------------------
const RandomForestClassifier_ = sken(:RandomForestClassifier)
@sk_clf mutable struct RandomForestClassifier <: MMI.Probabilistic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "gini"::(_ in ("gini","entropy", "log_loss"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    ## TODO: Remove the "auto" option in python sklearn v1.3
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
    class_weight::Any              = nothing
    ccp_alpha::Float64             =0.0::(_ ≥ 0)
    max_samples::Union{Nothing,Float64,Int} =
        nothing::(_ === nothing || (_ ≥ 0 && (_ isa Integer || _ ≤ 1)))
end
MMI.fitted_params(m::RandomForestClassifier, (f, _, _)) = (
    estimators            = f.estimators_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    n_features            = f.n_features_in_,
    n_outputs             = f.n_outputs_,
    feature_importances   = f.feature_importances_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
meta(RandomForestClassifier,
    input   = Table(Count,Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )

const ENSEMBLE_REG = Union{Type{<:AdaBoostRegressor}, Type{<:BaggingRegressor}, Type{<:GradientBoostingRegressor}}

MMI.input_scitype(::ENSEMBLE_REG)  = Table(Continuous)
MMI.target_scitype(::ENSEMBLE_REG) = AbstractVector{Continuous}

# ============================================================================
const ExtraTreesRegressor_ = sken(:ExtraTreesRegressor)
@sk_reg mutable struct ExtraTreesRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_>0)
    criterion::String              = "squared_error"::(_ in ("squared_error","absolute_error", "friedman_mse", "poisson"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64}  = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}   = 1::(_ > 0)
    min_weight_fraction_leaf::Float64      = 0.0::(_ ≥ 0)
    ## TODO: Remove the "auto" option in python sklearn v1.3
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
end
MMI.fitted_params(m::ExtraTreesRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    feature_importances = f.feature_importances_,
    n_features          = f.n_features_in_,
    n_outputs           = f.n_outputs_,
    oob_score           = m.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = m.oob_score ? f.oob_prediction_ : nothing,
    )
meta(ExtraTreesRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    weights = false
    )

"""
$(MMI.doc_header(ExtraTreesRegressor))

Extra trees regressor, fits a number of randomized decision trees on
various sub-samples of the dataset and uses averaging to improve the
predictive accuracy and control over-fitting.

"""
ExtraTreesRegressor

# ----------------------------------------------------------------------------
const ExtraTreesClassifier_ = sken(:ExtraTreesClassifier)
@sk_clf mutable struct ExtraTreesClassifier <: MMI.Probabilistic
    n_estimators::Int              = 100::(_>0)
    criterion::String              = "gini"::(_ in ("gini", "entropy", "log_loss"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64}  = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}   = 1::(_ > 0)
    min_weight_fraction_leaf::Float64      = 0.0::(_ ≥ 0)
    ## TODO: Remove the "auto" option in python sklearn v1.3
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
    class_weight::Any              = nothing
end
MMI.fitted_params(m::ExtraTreesClassifier, (f, _, _)) = (
    estimators            = f.estimators_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    feature_importances   = f.feature_importances_,
    n_features            = f.n_features_in_,
    n_outputs             = f.n_outputs_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing,
    )
meta(ExtraTreesClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )

"""
$(MMI.doc_header(ExtraTreesClassifier))

Extra trees classifier, fits a number of randomized decision trees on
various sub-samples of the dataset and uses averaging to improve the
predictive accuracy and control over-fitting.

"""
ExtraTreesClassifier
