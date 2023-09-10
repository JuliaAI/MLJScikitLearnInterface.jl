const SVMLinearClassifier_ = sksv(:LinearSVC)
@sk_clf mutable struct SVMLinearClassifier <: MMI.Deterministic
    penalty::String = "l2"::(_ in ("l1","l2"))
    loss::String    = "squared_hinge"::(_ in ("hinge", "squared_hinge"))
    dual::Bool      = true
    tol::Float64    = 1e-4::(_ > 0)
    C::Float64      = 1.0::(_ > 0)
    multi_class::String        = "ovr"::(_ in ("crammer_singer", "ovr"))
    fit_intercept::Bool        = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    random_state::Any          = nothing
    max_iter::Int              = 1000::(_ > 0)
end
MMI.fitted_params(m::SVMLinearClassifier, (f, _, _)) = (
    coef      = pyconvert(Array, f.coef_),
    intercept = pyconvert(Array, f.intercept_),
    classes   = pyconvert(Array, f.classes_),
    n_iter    = pyconvert(Int, f.n_iter_)
    )
meta(SVMLinearClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    human_name   = "linear support vector classifier"
    )

# ----------------------------------------------------------------------------
const SVMClassifier_ = sksv(:SVC)
@sk_clf mutable struct SVMClassifier <: MMI.Deterministic
    C::Float64          = 1.0::(_ > 0)
    kernel::Union{String,Function}  = "rbf"::(_ isa Function || _ in ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    degree::Int         = 3::(_ > 0)
    gamma::Union{Float64, String}    = "scale"::((_ isa Float64 && _ > 0) || _ in ("scale", "auto"))
    coef0::Float64      = 0.0
    shrinking::Bool     = true
    tol::Float64        = 1e-3::(_ > 0)
    cache_size::Int     = 200::(_ > 0)
    max_iter::Int       = -1
    decision_function_shape::String = "ovr"::(_ in ("ovo", "ovr"))
    random_state        = nothing
end
MMI.fitted_params(m::SVMClassifier, (f, _, _)) = (
    support         = pyconvert(Array, f.support_),
    support_vectors = pyconvert(Array, f.support_vectors_),
    n_support       = pyconvert(Array, f.n_support_),
    dual_coef       = pyconvert(Array, f.dual_coef_),
    coef            = m.kernel == "linear" ? pyconvert(Array, f.coef_) : nothing,
    intercept       = pyconvert(Array, f.intercept_),
    fit_status      = pyconvert(Int, f.fit_status_),
    classes         = pyconvert(Array, f.classes_)
    # probA = f.probA_,
    # probB = f.probB_,
    )
meta(SVMClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    human_name   = "C-support vector classifier"
    )

# ============================================================================
const SVMLinearRegressor_ = sksv(:LinearSVR)
@sk_reg mutable struct SVMLinearRegressor <: MMI.Deterministic
    epsilon::Float64 = 0.0::(_ ≥ 0)
    tol::Float64     = 1e-4::(_ > 0)
    C::Float64       = 1.0::(_ > 0)
    loss::String      = "epsilon_insensitive"::(_ in ("epsilon_insensitive", "squared_epsilon_insensitive"))
    fit_intercept::Bool        = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    dual::Bool                 = true
    random_state::Any          = nothing
    max_iter::Int              = 1000::(_ > 0)
end
MMI.fitted_params(m::SVMLinearRegressor, (f, _, _)) = (
    coef = pyconvert(Array, f.coef_),
    intercept = pyconvert(Array, f.intercept_),
    n_iter = pyconvert(Int, f.n_iter_)
    )
meta(SVMLinearRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    human_name   = "linear support vector regressor"
    )

# ----------------------------------------------------------------------------
const SVMRegressor_ = sksv(:SVR)
@sk_reg mutable struct SVMRegressor <: MMI.Deterministic
    kernel::Union{String,Function} = "rbf"::(_ isa Function || _ in ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    degree::Int      = 3::(_ > 0)
    gamma::Union{Float64,String}   = "scale"::((_ isa Float64 && _ > 0) || _ in ("scale", "auto"))
    coef0::Float64   = 0.0
    tol::Float64     = 1e-3::(_ > 0)
    C::Float64       = 1.0::(_ > 0)
    epsilon::Float64 = 0.1::(_ ≥ 0)
    shrinking        = true
    cache_size::Int  = 200::(_ > 0)
    max_iter::Int    = -1
end
MMI.fitted_params(m::SVMRegressor, (f, _, _)) = (
    support         = pyconvert(Array, f.support_),
    support_vectors = pyconvert(Array, f.support_vectors_),
    dual_coef       = pyconvert(Array, f.dual_coef_),
    coef            = m.kernel == "linear" ? pyconvert(Array, f.coef_) : nothing,
    intercept       = pyconvert(Array, f.intercept_),
    fit_status      = pyconvert(Int, f.fit_status_),
    n_iter          = pyconvert(Int, f.n_iter_)
    )
meta(SVMRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    human_name   = "epsilon-support vector regressor"
    )

# ============================================================================
const SVMNuClassifier_ = sksv(:NuSVC)
@sk_clf mutable struct SVMNuClassifier <: MMI.Deterministic
    nu::Float64 = 0.5::(0 < _ <= 1)
    kernel::Union{String,Function}  = "rbf"::(_ isa Function || _ in ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    degree::Int         = 3::(_ > 0)
    gamma::Union{Float64,String}    = "scale"::((_ isa Float64 && _ > 0) || _ in ("scale", "auto"))
    coef0::Float64      = 0.0
    shrinking::Bool     = true
    # probability
    tol::Float64        = 1e-3::(_ > 0)
    cache_size::Int     = 200::(_ > 0)
    max_iter::Int       = -1
    decision_function_shape::String = "ovr"::(_ in ("ovo", "ovr"))
    # break_ties::Bool    = false # XXX only >= 0.22
    random_state        = nothing
end
MMI.fitted_params(m::SVMNuClassifier, (f, _, _)) = (
    support         = pyconvert(Array, f.support_),
    support_vectors = pyconvert(Array, f.support_vectors_),
    n_support       = pyconvert(Array, f.n_support_),
    dual_coef       = pyconvert(Array, f.dual_coef_),
    coef            = m.kernel == "linear" ? pyconvert(Array, f.coef_) : nothing,
    intercept       = pyconvert(Array, f.intercept_),
    fit_status      = pyconvert(Int, f.fit_status_),
    classes         = pyconvert(Array, f.classes_),
    n_iter          = pyconvert(Int, f.n_iter_)
    # probA = f.probA_,
    # probB = f.probB_,
    )
meta(SVMNuClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    human_name   = "nu-support vector classifier"
    )

# ============================================================================
const SVMNuRegressor_ = sksv(:NuSVR)
@sk_reg mutable struct SVMNuRegressor <: MMI.Deterministic
    nu::Float64 = 0.5::(0 < _ <= 1)
    C::Float64       = 1.0::(_ > 0)
    kernel::Union{String,Function} = "rbf"::(_ isa Function || _ in ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    degree::Int      = 3::(_ > 0)
    gamma::Union{Float64,String}   = "scale"::((_ isa Float64 && _ > 0) || _ in ("scale", "auto"))
    coef0::Float64   = 0.0
    shrinking        = true
    tol::Float64     = 1e-3::(_ > 0)
    cache_size::Int  = 200::(_ > 0)
    max_iter::Int    = -1
end
MMI.fitted_params(m::SVMNuRegressor, (f, _, _)) = (
    support         = pyconvert(Array, f.support_),
    support_vectors = pyconvert(Array, f.support_vectors_),
    dual_coef       = pyconvert(Array, f.dual_coef_),
    coef            = m.kernel == "linear" ? pyconvert(Array, f.coef_) : nothing,
    intercept       = pyconvert(Array, f.intercept_),
    n_iter          = pyconvert(Int, f.n_iter_)
    )
meta(SVMNuRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    human_name   = "nu-support vector regressor"
    )
