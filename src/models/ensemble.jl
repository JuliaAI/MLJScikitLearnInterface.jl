const AdaBoostRegressor_ = sken(:AdaBoostRegressor)
@sk_reg mutable struct AdaBoostRegressor <: MMI.Deterministic
    estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    loss::String           = "linear"::(_ in ("linear","square","exponential"))
    random_state::Any      = nothing
end
MMI.fitted_params(model::AdaBoostRegressor, (f, _, _)) = (
    estimator            = f.estimator_,
    estimators           = f.estimators_,
    estimator_weights    = pyconvert(Array, f.estimator_weights_),
    estimator_errors     = pyconvert(Array, f.estimator_errors_),
    feature_importances  = pyconvert(Array, f.feature_importances_)
    )
add_human_name_trait(AdaBoostRegressor, "AdaBoost ensemble regression")
"""
$(MMI.doc_header(AdaBoostRegressor))

An AdaBoost regressor is a meta-estimator that begins by fitting 
a regressor on the original dataset and then fits additional 
copies of the regressor on the same dataset but where the weights 
of instances are adjusted according to the error of the current 
prediction. As such, subsequent regressors focus more on difficult 
cases.

This class implements the algorithm known as AdaBoost.R2.

"""
AdaBoostRegressor

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
    estimator         = f.estimator_,
    estimators        = f.estimators_,
    estimator_weights = pyconvert(Array, f.estimator_weights_),
    estimator_errors  = pyconvert(Array, f.estimator_errors_),
    classes           = pyconvert(Array, f.classes_),
    n_classes         = pyconvert(Int, f.n_classes_)
    )
meta(AdaBoostClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    )
"""
$(MMI.doc_header(AdaBoostClassifier))

An AdaBoost  classifier is a meta-estimator that begins by fitting a 
classifier on the original dataset and then fits additional copies of 
the classifier on the same dataset but where the weights of incorrectly 
classified instances are adjusted such that subsequent classifiers 
focus more on difficult cases.

This class implements the algorithm known as AdaBoost-SAMME.

"""
AdaBoostClassifier

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
    estimator           = f.estimator_,
    estimators          = f.estimators_,
    estimators_samples  = f.estimators_samples_,
    estimators_features = f.estimators_features_,
    oob_score           = model.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
    oob_prediction      = model.oob_score ? pyconvert(Array, f.oob_prediction_) : nothing
    )
add_human_name_trait(BaggingRegressor, "bagging ensemble regressor")
"""
$(MMI.doc_header(BaggingRegressor))

A Bagging regressor is an ensemble meta-estimator that fits base 
regressors each on random subsets of the original dataset and then 
aggregate their individual predictions (either by voting or by 
averaging) to form a final prediction. Such a meta-estimator can 
typically be used as a way to reduce the variance of a black-box 
estimator (e.g., a decision tree), by introducing randomization 
into its construction procedure and then making an ensemble out 
of it.

"""
BaggingRegressor

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
    estimator             = f.estimator_,
    estimators            = f.estimators_,
    estimators_samples    = f.estimators_samples_,
    estimators_features   = f.estimators_features_,
    classes               = pyconvert(Array, f.classes_),
    n_classes             = pyconvert(Int, f.n_classes_),
    oob_score             = m.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
meta(BaggingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "bagging ensemble classifier"
    )
"""
$(MMI.doc_header(BaggingClassifier))

A Bagging classifier is an ensemble meta-estimator that fits base 
classifiers each on random subsets of the original dataset and then 
aggregate their individual predictions (either by voting or by 
averaging) to form a final prediction. Such a meta-estimator can 
typically be used as a way to reduce the variance of a black-box 
estimator (e.g., a decision tree), by introducing randomization into 
its construction procedure and then making an ensemble out of it.

"""
BaggingClassifier

# ============================================================================
const ExtraTreesRegressor_ = sken(:ExtraTreesRegressor)
@sk_reg mutable struct ExtraTreesRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_>0)
    criterion::String              = "squared_error"::(_ in ("squared_error","absolute_error", "friedman_mse", "poisson"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64}  = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}   = 1::(_ > 0)
    min_weight_fraction_leaf::Float64      = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = 1.0::(_ === nothing || (isa(_, String) && (_ in ("sqrt","log2"))) || (_ isa Number && _ > 0))
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
    estimator           = f.estimator_,
    estimators          = f.estimators_,
    feature_importances = pyconvert(Array, f.feature_importances_),
    n_features          = pyconvert(Int, f.n_features_in_),
    n_outputs           = pyconvert(Int, f.n_outputs_),
    oob_score           = m.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
    oob_prediction      = m.oob_score ? pyconvert(Array, f.oob_prediction_) : nothing,
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
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("sqrt","log2"))) || (_ isa Number && _ > 0))
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
    estimator             = f.estimator_,
    estimators            = f.estimators_,
    classes               = pyconvert(Array, f.classes_),
    n_classes             = pyconvert(Int, f.n_classes_),
    feature_importances   = pyconvert(Array, f.feature_importances_),
    n_features            = pyconvert(Int, f.n_features_in_),
    n_outputs             = pyconvert(Int, f.n_outputs_),
    oob_score             = m.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
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
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || (_ isa Number && _ > 0))
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
    feature_importances = pyconvert(Array, f.feature_importances_),
    train_score         = pyconvert(Array, f.train_score_),
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? pyconvert(Array, f.oob_improvement_) : nothing
    )
add_human_name_trait(GradientBoostingRegressor, "gradient boosting ensemble regression")
"""
$(MMI.doc_header(GradientBoostingRegressor))

This estimator builds an additive model in a forward stage-wise fashion; 
it allows for the optimization of arbitrary differentiable loss functions. 
In each stage a regression tree is fit on the negative gradient of the 
given loss function.

[`HistGradientBoostingRegressor`](@ref) is a much faster variant of this 
algorithm for intermediate datasets (`n_samples >= 10_000`).

"""
GradientBoostingRegressor

# ----------------------------------------------------------------------------
const GradientBoostingClassifier_ = sken(:GradientBoostingClassifier)
@sk_clf mutable struct GradientBoostingClassifier <: MMI.Probabilistic
    loss::String                    = "log_loss"::(_ in ("log_loss","exponential"))
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
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || (_ isa Number && _ > 0))
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
    feature_importances = pyconvert(Array, f.feature_importances_),
    train_score         = pyconvert(Array, f.train_score_),
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? pyconvert(Array, f.oob_improvement_) : nothing
    )
meta(GradientBoostingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )
"""
$(MMI.doc_header(GradientBoostingClassifier))

This algorithm builds an additive model in a forward stage-wise fashion; 
it allows for the optimization of arbitrary differentiable loss functions. 
In each stage `n_classes_` regression trees are fit on the negative gradient 
of the loss function, e.g. binary or multiclass log loss. Binary 
classification is a special case where only a single regression tree is induced.

[`HistGradientBoostingClassifier`](@ref) is a much faster variant of this 
algorithm for intermediate datasets (`n_samples >= 10_000`).

"""
GradientBoostingClassifier

# ============================================================================
const RandomForestRegressor_ = sken(:RandomForestRegressor)
@sk_reg mutable struct RandomForestRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "squared_error"::(_ in ("squared_error","absolute_error", "friedman_mse", "poisson"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = 1.0::(_ === nothing || (isa(_, String) && (_ in ("sqrt","log2"))) || (_ isa Number && _ > 0))
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
    estimator           = f.estimator_,
    estimators          = f.estimators_,
    feature_importances = pyconvert(Array, f.feature_importances_),
    n_features          = pyconvert(Int, f.n_features_in_),
    n_outputs           = pyconvert(Int, f.n_outputs_),
    oob_score           = model.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
    oob_prediction      = model.oob_score ? pyconvert(Array, f.oob_prediction_) : nothing
    )
meta(RandomForestRegressor,
    input   = Table(Count,Continuous),
    target  = AbstractVector{Continuous},
    weights = false
    )
"""
$(MMI.doc_header(RandomForestRegressor))

A random forest is a meta estimator that fits a number of 
classifying decision trees on various sub-samples of the 
dataset and uses averaging to improve the predictive accuracy 
and control over-fitting. The sub-sample size is controlled 
with the `max_samples` parameter if `bootstrap=True` (default), 
otherwise the whole dataset is used to build each tree.

"""
RandomForestRegressor

# ----------------------------------------------------------------------------
const RandomForestClassifier_ = sken(:RandomForestClassifier)
@sk_clf mutable struct RandomForestClassifier <: MMI.Probabilistic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "gini"::(_ in ("gini","entropy", "log_loss"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "sqrt"::(_ === nothing || (isa(_, String) && (_ in ("sqrt","log2"))) || (_ isa Number && _ > 0))
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
    estimator             = f.estimator_,
    estimators            = f.estimators_,
    classes               = pyconvert(Array, f.classes_),
    n_classes             = pyconvert(Int, f.n_classes_),
    n_features            = pyconvert(Int, f.n_features_in_),
    n_outputs             = pyconvert(Int, f.n_outputs_),
    feature_importances   = pyconvert(Array, f.feature_importances_),
    oob_score             = m.oob_score ? pyconvert(Float64, f.oob_score_) : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
meta(RandomForestClassifier,
    input   = Table(Count,Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )
"""
$(MMI.doc_header(RandomForestClassifier))

A random forest is a meta estimator that fits a number of 
classifying decision trees on various sub-samples of the 
dataset and uses averaging to improve the predictive accuracy 
and control over-fitting. The sub-sample size is controlled 
with the `max_samples` parameter if `bootstrap=True` (default), 
otherwise the whole dataset is used to build each tree.

"""
RandomForestClassifier

# ============================================================================
const HistGradientBoostingRegressor_ = sken(:HistGradientBoostingRegressor)
@sk_reg mutable struct HistGradientBoostingRegressor <: MMI.Deterministic
    loss::String                    = "squared_error"::(_ in ("squared_error","absolute_error","gamma","poisson", "quantile"))
    quantile::Option{Float64}       = nothing::(_===nothing || 0<_>1)
    learning_rate::Float64          = 0.1::(_>0)
    max_iter::Int                   = 100::(_>0)
    max_leaf_nodes::Option{Int}     = 31::(_===nothing || _>0)
    max_depth::Option{Int}          = nothing::(_===nothing || _>0)
    min_samples_leaf::Union{Int,Float64} = 20::(_>0)
    l2_regularization::Float64      = 0.0
    max_bins::Int                   = 255
    categorical_features::Option{Vector} = nothing
    monotonic_cst::Option{Union{Vector, Dict}} = nothing
    # interaction_cst
    warm_start::Bool                = false
    early_stopping::Union{String, Bool} = "auto"::(_ in ("auto", true, false))
    scoring::String                 = "loss"
    validation_fraction::Option{Union{Int, Float64}} = 0.1::(_===nothing || _≥0)
    n_iter_no_change::Option{Int}  = 10::(_===nothing || _>0)
    tol::Float64                   = 1e-7::(_>0)
    random_state::Any              = nothing
end
MMI.fitted_params(m::HistGradientBoostingRegressor, (f, _, _)) = (
    do_early_stopping   = pyconvert(Bool, f.do_early_stopping_),
    n_iter              = pyconvert(Int, f.n_iter_),
    n_trees_per_iteration = pyconvert(Int, f.n_trees_per_iteration_),
    train_score         = pyconvert(Array, f.train_score_),
    validation_score    = pyconvert(Array, f.validation_score_)
    )
add_human_name_trait(HistGradientBoostingRegressor, "gradient boosting ensemble regression")
"""
$(MMI.doc_header(HistGradientBoostingRegressor))

This estimator builds an additive model in a forward stage-wise fashion; 
it allows for the optimization of arbitrary differentiable loss functions. 
In each stage a regression tree is fit on the negative gradient of the 
given loss function.

[`HistGradientBoostingRegressor`](@ref) is a much faster variant of this 
algorithm for intermediate datasets (`n_samples >= 10_000`).

"""
HistGradientBoostingRegressor

# ----------------------------------------------------------------------------
const HistGradientBoostingClassifier_ = sken(:HistGradientBoostingClassifier)
@sk_clf mutable struct HistGradientBoostingClassifier <: MMI.Probabilistic
    loss::String                    = "log_loss"::(_ in ("log_loss",))
    learning_rate::Float64          = 0.1::(_>0)
    max_iter::Int                   = 100::(_>0)
    max_leaf_nodes::Option{Int}     = 31::(_===nothing || _>0)
    max_depth::Option{Int}          = nothing::(_===nothing || _>0)
    min_samples_leaf::Union{Int,Float64} = 20::(_>0)
    l2_regularization::Float64      = 0.0
    max_bins::Int                   = 255
    categorical_features::Option{Vector} = nothing
    monotonic_cst::Option{Union{Vector, Dict}} = nothing
    # interaction_cst
    warm_start::Bool                = false
    early_stopping::Union{String, Bool} = "auto"::(_ in ("auto",) || _ isa Bool)
    scoring::String                 = "loss"
    validation_fraction::Option{Union{Int, Float64}} = 0.1::(_===nothing || _≥0)
    n_iter_no_change::Option{Int}  = 10::(_===nothing || _>0)
    tol::Float64                   = 1e-7::(_>0)
    random_state::Any              = nothing
    class_weight::Any              = nothing
end
MMI.fitted_params(m::HistGradientBoostingClassifier, (f, _, _)) = (
    classes             = pyconvert(Array, f.classes_),
    do_early_stopping   = pyconvert(Bool, f.do_early_stopping_),
    n_iter              = pyconvert(Int, f.n_iter_),
    n_trees_per_iteration = pyconvert(Int, f.n_trees_per_iteration_),
    train_score         = pyconvert(Array, f.train_score_),
    validation_score    = pyconvert(Array, f.validation_score_)
    )
meta(HistGradientBoostingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )
"""
$(MMI.doc_header(HistGradientBoostingClassifier))

This algorithm builds an additive model in a forward stage-wise fashion; 
it allows for the optimization of arbitrary differentiable loss functions. 
In each stage `n_classes_` regression trees are fit on the negative gradient 
of the loss function, e.g. binary or multiclass log loss. Binary 
classification is a special case where only a single regression tree is induced.

[`HistGradientBoostingClassifier`](@ref) is a much faster variant of this 
algorithm for intermediate datasets (`n_samples >= 10_000`).

"""
HistGradientBoostingClassifier

# ----------------------------------------------------------------------------
const ENSEMBLE_REG = Union{Type{<:AdaBoostRegressor}, 
                           Type{<:BaggingRegressor}, 
                           Type{<:ExtraTreesRegressor},
                           Type{<:GradientBoostingRegressor},
                           Type{<:HistGradientBoostingRegressor}}

MMI.input_scitype(::ENSEMBLE_REG)  = Table(Continuous)
MMI.target_scitype(::ENSEMBLE_REG) = AbstractVector{Continuous}
