const LogisticClassifier_ = sklm(:LogisticRegression)
@sk_clf mutable struct LogisticClassifier <: MMI.Probabilistic
    penalty::String            = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    dual::Bool                 = false
    tol::Float64               = 1e-4::(_ > 0)
    C::Float64                 = 1.0::(_ > 0)
    fit_intercept::Bool        = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    class_weight::Any          = nothing
    random_state::Any          = nothing
    solver::String             = "lbfgs"::(_ in ("lbfgs", "newton-cg", "liblinear", "sag", "saga"))
    max_iter::Int              = 100::(_ > 0)
    multi_class::String        = "auto"::(_ in ("ovr", "multinomial", "auto"))
    verbose::Int               = 0
    warm_start::Bool           = false
    n_jobs::Option{Int}        = nothing
    l1_ratio::Option{Float64}  = nothing::(_ === nothing || 0 ≤ _ ≤ 1)
end
MMI.fitted_params(m::LogisticClassifier, (f, _, _)) = (
    classes   = pyconvert(Array, f.classes_),
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta(LogisticClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name  = "logistic regression classifier"
    )
@sk_feature_importances LogisticClassifier

# ============================================================================
const LogisticCVClassifier_ = sklm(:LogisticRegressionCV)
@sk_clf mutable struct LogisticCVClassifier <: MMI.Probabilistic
    Cs::Union{Int,AbstractVector{Float64}} = 10::((_ isa Int && _ > 0) || all(_ .> 0))
    fit_intercept::Bool        = true
    cv::Any                    = 5
    dual::Bool                 = false
    penalty::String            = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    scoring::Any               = nothing
    solver::String             = "lbfgs"::(_ in ("lbfgs", "newton-cg", "liblinear", "sag", "saga"))
    tol::Float64               = 1e-4::(_ > 0)
    max_iter::Int              = 100::(_ > 0)
    class_weight::Any          = nothing
    n_jobs::Option{Int}        = nothing
    verbose::Int               = 0
    refit::Bool                = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    multi_class::String        = "auto"::(_ in ("ovr", "multinomial", "auto"))
    random_state::Any          = nothing
    l1_ratios::Option{AbstractVector{Float64}}=nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
end
MMI.fitted_params(m::LogisticCVClassifier, (f, _, _)) = (
    classes     = pyconvert(Array, f.classes_),
    coef        = pyconvert(Array, f.coef_),
    intercept   = m.fit_intercept ? pyconvert(Array, f.intercept_) : nothing,
    Cs          = pyconvert(Array, f.Cs_),
    l1_ratios   = ifelse(m.penalty == "elasticnet", f.l1_ratios_, nothing),
    coefs_paths = pyconvert(Array, f.coefs_paths_),
    scores      = f.scores_,
    C           = pyconvert(Array, f.C_),
    l1_ratio    = pyconvert(Array, f.l1_ratio_)
    )
meta(LogisticCVClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "logistic regression classifier $CV"
    )
@sk_feature_importances LogisticCVClassifier

# ============================================================================
const PassiveAggressiveClassifier_ = sklm(:PassiveAggressiveClassifier)
@sk_clf mutable struct PassiveAggressiveClassifier <: MMI.Deterministic
    C::Float64            = 1.0::(_ > 0)
    fit_intercept::Bool   = true
    max_iter::Int         = 100::(_ > 0)
    tol::Float64          = 1e-3::(_ > 0)
    early_stopping::Bool  = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int = 5::(_ > 0)
    shuffle::Bool         = true
    verbose::Int          = 0
    loss::String          = "hinge"::(_ in ("hinge", "squared_hinge"))
    n_jobs::Option{Int}   = nothing
    random_state::Any     = 0
    warm_start::Bool      = false
    class_weight::Any     = nothing
    average::Bool         = false
end
MMI.fitted_params(m::PassiveAggressiveClassifier, (f, _, _)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta(PassiveAggressiveClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "passive aggressive classifier"
    )
@sk_feature_importances PassiveAggressiveClassifier

# ============================================================================
const PerceptronClassifier_ = sklm(:Perceptron)
@sk_clf mutable struct PerceptronClassifier <: MMI.Deterministic
    penalty::Option{String} = nothing::(_ === nothing || _ in ("l2", "l1", "elasticnet"))
    alpha::Float64          = 1e-4::(_ > 0)
    fit_intercept::Bool     = true
    max_iter::Int           = 1_000::(_ > 0)
    tol::Option{Float64}    = 1e-3
    shuffle::Bool           = true
    verbose::Int            = 0
    eta0::Float64           = 1.0::(_ > 0)
    n_jobs::Option{Int}     = nothing
    random_state::Any       = 0
    early_stopping::Bool    = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int   = 5::(_ > 0)
    class_weight::Any       = nothing
    warm_start::Bool        = false
end
MMI.fitted_params(m::PerceptronClassifier, (f, _, _)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta(PerceptronClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    )
@sk_feature_importances PerceptronClassifier

# ============================================================================
const RidgeClassifier_ = sklm(:RidgeClassifier)
@sk_clf mutable struct RidgeClassifier <: MMI.Deterministic
    alpha::Float64        = 1.0
    fit_intercept::Bool   = true
    copy_X::Bool          = true
    max_iter::Option{Int} = nothing::(_ === nothing || _ > 0)
    tol::Float64          = 1e-3::(arg>0)
    class_weight::Any     = nothing
    solver::String        = "auto"::(arg in ("auto","svd","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any     = nothing
end
MMI.fitted_params(m::RidgeClassifier, (f, _, _)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta(RidgeClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "ridge regression classifier"
    )
@sk_feature_importances RidgeClassifier

# ============================================================================
const RidgeCVClassifier_ = sklm(:RidgeClassifierCV)
@sk_clf mutable struct RidgeCVClassifier <: MMI.Deterministic
    alphas::AbstractArray{Float64} = [0.1,1.0,10.0]::(all(0 .≤ _))
    fit_intercept::Bool   = true
    scoring::Any          = nothing
    cv::Int               = 5
    class_weight::Any     = nothing
    store_cv_values::Bool = false
end
MMI.fitted_params(m::RidgeCVClassifier, (f, _, _)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta(RidgeCVClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "ridge regression classifier $CV"
    )
@sk_feature_importances RidgeCVClassifier

# ============================================================================
const SGDClassifier_ = sklm(:SGDClassifier)
@sk_clf mutable struct SGDClassifier <: MMI.Deterministic
    loss::String          = "hinge"::(_ in ("hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"))
    penalty::String       = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    alpha::Float64        = 1e-4::(_ > 0)
    l1_ratio::Float64     = 0.15::(0 ≤ _ ≤ 1)
    fit_intercept::Bool   = true
    max_iter::Int         = 1_000::(_ > 0)
    tol::Option{Float64}  = 1e-3::(_ === nothing || _ > 0)
    shuffle::Bool         = true
    verbose::Int          = 0
    epsilon::Float64      = 0.1::(_ > 0)
    n_jobs::Option{Int}   = nothing
    random_state::Any     = nothing
    learning_rate::String = "optimal"::(_ in ("constant", "optimal", "invscaling", "adaptive"))
    eta0::Float64         = 0.0::(_ ≥ 0)
    power_t::Float64      = 0.5::(_ > 0)
    early_stopping::Bool  = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int = 5::(_ > 0)
    class_weight::Any     = nothing
    warm_start::Bool      = false
    average::Bool         = false
end
const ProbabilisticSGDClassifier_ = sklm(:SGDClassifier)
@sk_clf mutable struct ProbabilisticSGDClassifier <: MMI.Probabilistic
    loss::String          = "log_loss"::(_ in ("log_loss", "modified_huber")) # only those -> predict proba
    penalty::String       = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    alpha::Float64        = 1e-4::(_ > 0)
    l1_ratio::Float64     = 0.15::(0 ≤ _ ≤ 1)
    fit_intercept::Bool   = true
    max_iter::Int         = 1_000::(_ > 0)
    tol::Option{Float64}  = 1e-3::(_ === nothing || _ > 0)
    shuffle::Bool         = true
    verbose::Int          = 0
    epsilon::Float64      = 0.1::(_ > 0)
    n_jobs::Option{Int}   = nothing
    random_state::Any     = nothing
    learning_rate::String = "optimal"::(_ in ("constant", "optimal", "invscaling", "adaptive"))
    eta0::Float64         = 0.0::(_ ≥ 0)
    power_t::Float64      = 0.5::(_ > 0)
    early_stopping::Bool  = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int = 5::(_ > 0)
    class_weight::Any     = nothing
    warm_start::Bool      = false
    average::Bool         = false
end
MMI.fitted_params(m::SGDClassifier, (f,_,_)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
# duplication to avoid ambiguity that julia doesn't like
MMI.fitted_params(m::ProbabilisticSGDClassifier, (f,_,_)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
meta.((SGDClassifier, ProbabilisticSGDClassifier),
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false
    )
@sk_feature_importances SGDClassifier
