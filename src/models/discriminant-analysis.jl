const BayesianLDA_ = skda(:LinearDiscriminantAnalysis)
@sk_clf mutable struct BayesianLDA <: MMI.Probabilistic
    solver::String                   = "svd"::(_ in ("svd", "lsqr", "eigen"))
    shrinkage::Union{Nothing,String,Float64} = nothing::(_ === nothing || _ == "auto" || 0 < _ < 1)
    priors::Option{AbstractVector}   = nothing
    n_components::Option{Int}        = nothing
    store_covariance::Bool           = false
    tol::Float64                     = 1e-4::(_ > 0)
    covariance_estimator::Any        = nothing
end
MMI.fitted_params(m::BayesianLDA, (f, _, _)) = (
    coef       = pyconvert(Array, f.coef_),
    intercept  = pyconvert(Array, f.intercept_),
    covariance = m.store_covariance ? pyconvert(Array, f.covariance_) : nothing,
    explained_variance_ratio = pyconvert(Array, f.explained_variance_ratio_),
    means      = pyconvert(Array, f.means_),
    priors     = pyconvert(Array, f.priors_),
    scalings   = pyconvert(Array, f.scalings_),
    xbar       = pyconvert(Array, f.xbar_),
    classes    = pyconvert(Array, f.classes_)
    )
meta(BayesianLDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "Bayesian linear discriminant analysis"
    )
@sk_feature_importances BayesianLDA

# ============================================================================
const BayesianQDA_ = skda(:QuadraticDiscriminantAnalysis)
@sk_clf mutable struct BayesianQDA <: MMI.Probabilistic
    priors::Option{AbstractVector} = nothing
    reg_param::Float64             = 0.0::(_ ≥ 0)
    store_covariance::Bool         = false
    tol::Float64                   = 1e-4::(_ > 0)
end
MMI.fitted_params(m::BayesianQDA, (f, _, _)) = (
    covariance = m.store_covariance ? pyconvert(Array, f.covariance_) : nothing,
    means      = pyconvert(Array, f.means_),
    priors     = pyconvert(Array, f.priors_),
    rotations  = pyconvert(Array, f.rotations_),
    scalings   = pyconvert(Array, f.scalings_),
    )
meta(BayesianQDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "Bayesian quadratic discriminant analysis"
    )
