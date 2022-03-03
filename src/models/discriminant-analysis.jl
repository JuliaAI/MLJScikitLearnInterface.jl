const BayesianLDA_ = skda(:LinearDiscriminantAnalysis)
@sk_clf mutable struct BayesianLDA <: MMI.Probabilistic
    solver::String                   = "svd"::(_ in ("svd", "lsqr", "eigen"))
    shrinkage::Union{Nothing,String,Float64} = nothing::(_ === nothing || _ == "auto" || 0 < _ < 1)
    priors::Option{AbstractVector}   = nothing
    n_components::Option{Int}        = nothing
    store_covariance::Bool           = false
    tol::Float64                     = 1e-4::(_ > 0)
end
MMI.fitted_params(m::BayesianLDA, (f, _, _)) = (
    coef       = f.coef_,
    intercept  = f.intercept_,
    covariance = m.store_covariance ? f.covariance_ : nothing,
    means      = f.means_,
    priors     = f.priors_,
    scalings   = f.scalings_,
    xbar       = f.xbar_,
    classes    = f.classes_,
    explained_variance_ratio = f.explained_variance_ratio_
    )
meta(BayesianLDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "Bayesian linear discriminant analysis"
    )

# ============================================================================
const BayesianQDA_ = skda(:QuadraticDiscriminantAnalysis)
@sk_clf mutable struct BayesianQDA <: MMI.Probabilistic
    priors::Option{AbstractVector} = nothing
    reg_param::Float64             = 0.0::(_ ≥ 0)
    store_covariance::Bool         = false
    tol::Float64                   = 1e-4::(_ > 0)
end
MMI.fitted_params(m::BayesianQDA, (f, _, _)) = (
    covariance = m.store_covariance ? f.covariance_ : nothing,
    means      = f.means_,
    priors     = f.priors_,
    rotations  = f.rotations_,
    scalings   = f.scalings_
    )
meta(BayesianQDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "Bayesian quadratic discriminant analysis"
    )
