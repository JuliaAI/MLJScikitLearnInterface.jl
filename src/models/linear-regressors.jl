const ARDRegressor_ = sklm(:ARDRegression)
@sk_reg mutable struct ARDRegressor <: MMI.Deterministic
    # TODO: rename `n_iter` to `max_iter` in v1.5
    n_iter::Int               = 300::(_ > 0)
    tol::Float64              = 1e-3::(_ > 0)
    alpha_1::Float64          = 1e-6::(_ > 0)
    alpha_2::Float64          = 1e-6::(_ > 0)
    lambda_1::Float64         = 1e-6::(_ > 0)
    lambda_2::Float64         = 1e-6::(_ > 0)
    compute_score::Bool       = false
    threshold_lambda::Float64 = 1e4::(_ > 0)
    fit_intercept::Bool       = true
    copy_X::Bool              = true
    verbose::Bool             = false
end
MMI.fitted_params(model::ARDRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    lambda    = fitresult.lambda_,
    sigma     = fitresult.sigma_,
    scores    = fitresult.scores_
    )
add_human_name_trait(ARDRegressor, "Bayesian ARD regressor")

# =============================================================================
const BayesianRidgeRegressor_ = sklm(:BayesianRidge)
@sk_reg mutable struct BayesianRidgeRegressor <: MMI.Deterministic
    # TODO: rename `n_iter` to `max_iter` in v1.5
    n_iter::Int         = 300::(_ ≥ 1)
    tol::Float64        = 1e-3::(_ > 0)
    alpha_1::Float64    = 1e-6::(_ > 0)
    alpha_2::Float64    = 1e-6::(_ > 0)
    lambda_1::Float64   = 1e-6::(_ > 0)
    lambda_2::Float64   = 1e-6::(_ > 0)
    compute_score::Bool = false
    fit_intercept::Bool = true
    copy_X::Bool        = true
    verbose::Bool       = false
end
MMI.fitted_params(model::BayesianRidgeRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    lambda    = fitresult.lambda_,
    sigma     = fitresult.sigma_,
    scores    = fitresult.scores_
    )
add_human_name_trait(BayesianRidgeRegressor, "Bayesian ridge regressor")

# =============================================================================
const ElasticNetRegressor_ = sklm(:ElasticNet)
@sk_reg mutable struct ElasticNetRegressor <: MMI.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)   # 0 is OLS
    l1_ratio::Float64   = 0.5::(0 ≤ _ ≤ 1)
    fit_intercept::Bool = true
    precompute::Union{Bool,AbstractMatrix} = false
    max_iter::Int       = 1_000::(_ ≥ 1)
    copy_X::Bool        = true
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    positive::Bool      = false
    random_state::Any   = nothing  # Int, random state, or nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::ElasticNetRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    )

# =============================================================================
const ElasticNetCVRegressor_ = sklm(:ElasticNetCV)
@sk_reg mutable struct ElasticNetCVRegressor <: MMI.Deterministic
    l1_ratio::Union{Float64,Vector{Float64}} = 0.5::(all(0 .≤ _ .≤ 1))
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _))
    fit_intercept::Bool = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    cv::Any             = 5 # can be Int, Nothing or an iterable / cv splitter
    copy_X::Bool        = true
    verbose::Union{Bool, Int}  = 0
    n_jobs::Option{Int} = nothing
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::ElasticNetCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    l1_ratio  = fitresult.l1_ratio_,
    mse_path  = fitresult.mse_path_,
    alphas    = fitresult.alphas_
    )
add_human_name_trait(ElasticNetCVRegressor, "elastic net regression $CV")

# =============================================================================
const HuberRegressor_ = sklm(:HuberRegressor)
@sk_reg mutable struct HuberRegressor <: MMI.Deterministic
    epsilon::Float64    = 1.35::(_ > 1.0)
    max_iter::Int       = 100::(_ > 0)
    alpha::Float64      = 1e-4::(_ > 0)
    warm_start::Bool    = false
    fit_intercept::Bool = true
    tol::Float64        = 1e-5::(_ > 0)
end
MMI.fitted_params(model::HuberRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    scale     = fitresult.scale_,
    outliers  = fitresult.outliers_
    )
add_human_name_trait(HuberRegressor, "Huber regressor")

# =============================================================================
const LarsRegressor_ = sklm(:Lars)
@sk_reg mutable struct LarsRegressor <: MMI.Deterministic
    fit_intercept::Bool      = true
    verbose::Union{Bool,Int} = false
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool          = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    n_nonzero_coefs::Int     = 500::(_ > 0)
    eps::Float64   = eps(Float64)::(_ > 0)
    copy_X::Bool   = true
    fit_path::Bool = true
end
MMI.fitted_params(model::LarsRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alphas    = fitresult.alphas_,
    active    = fitresult.active_,
    coef_path = fitresult.coef_path_
    )
add_human_name_trait(LarsRegressor, "least angle regressor (LARS)")

# =============================================================================
const LarsCVRegressor_ = sklm(:LarsCV)
@sk_reg mutable struct LarsCVRegressor <: MMI.Deterministic
    fit_intercept::Bool      = true
    verbose::Union{Bool,Int} = false
    max_iter::Int     = 500::(_ > 0)
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool   = false 
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    cv::Any           = 5
    max_n_alphas::Int = 1_000::(_ > 0)
    n_jobs::Option{Int} = nothing
    eps::Float64      = eps(Float64)::(_ > 0)
    copy_X::Bool      = true
end
MMI.fitted_params(model::LarsCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    alphas    = fitresult.alphas_,
    cv_alphas = fitresult.cv_alphas_,
    mse_path  = fitresult.mse_path_,
    coef_path = fitresult.coef_path_
    )
add_human_name_trait(LarsCVRegressor, "least angle regressor $CV")

# =============================================================================
const LassoRegressor_ = sklm(:Lasso)
@sk_reg mutable struct LassoRegressor <: MMI.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0) # should use alpha > 0 (or OLS)
    fit_intercept::Bool = true
    precompute::Union{Bool,AbstractMatrix} = false
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::LassoRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    )

# =============================================================================
const LassoCVRegressor_ = sklm(:LassoCV)
@sk_reg mutable struct LassoCVRegressor <: MMI.Deterministic
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _))
    fit_intercept::Bool = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    cv::Any             = 5
    verbose::Union{Bool,Int} = false
    n_jobs::Option{Int} = nothing
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::LassoCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    alphas    = fitresult.alphas_,
    mse_path  = fitresult.mse_path_,
    dual_gap  = fitresult.dual_gap_
    )
add_human_name_trait(LassoCVRegressor, "lasso regressor $CV")

# =============================================================================
const LassoLarsRegressor_ = sklm(:LassoLars)
@sk_reg mutable struct LassoLarsRegressor <: MMI.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0) # 0 should be OLS
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool     = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 500::(_ > 0)
    eps::Float64        = eps(Float64)::(_ > 0)
    copy_X::Bool        = true
    fit_path::Bool      = true
    positive::Any       = false
end
MMI.fitted_params(model::LassoLarsRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alphas    = fitresult.alphas_,
    active    = fitresult.active_,
    coef_path = fitresult.coef_path_
    )
add_human_name_trait(LassoLarsRegressor, "Lasso model fit with least angle regression (LARS)")

# =============================================================================
const LassoLarsCVRegressor_ = sklm(:LassoLarsCV)
@sk_reg mutable struct LassoLarsCVRegressor <: MMI.Deterministic
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    max_iter::Int       = 500::(_ > 0)
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool     = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    cv::Any             = 5
    max_n_alphas::Int   = 1_000::(_ > 0)
    n_jobs::Option{Int} = nothing
    eps::Float64        = eps(Float64)::(_ > 0.0)
    copy_X::Bool        = true
    positive::Any       = false
end
MMI.fitted_params(model::LassoLarsCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    coef_path = fitresult.coef_path_,
    alpha     = fitresult.alpha_,
    alphas    = fitresult.alphas_,
    cv_alphas = fitresult.cv_alphas_,
    mse_path  = fitresult.mse_path_
    )
add_human_name_trait(LassoLarsCVRegressor, "Lasso model fit with least angle "*
                     "regression (LARS) $CV")

# =============================================================================
const LassoLarsICRegressor_ = sklm(:LassoLarsIC)
@sk_reg mutable struct LassoLarsICRegressor <: MMI.Deterministic
    criterion::String   = "aic"::(_ in ("aic","bic"))
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool     = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 500::(_ > 0)
    eps::Float64        = eps(Float64)::(_ > 0.0)
    copy_X::Bool        = true
    positive::Any       = false
end
MMI.fitted_params(model::LassoLarsICRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_
    )
add_human_name_trait(LassoLarsICRegressor, "Lasso model with LARS using "*
                     "BIC or AIC for model selection")

# =============================================================================
const LinearRegressor_ = sklm(:LinearRegression)
@sk_reg mutable struct LinearRegressor <: MMI.Deterministic
    fit_intercept::Bool = true
    copy_X::Bool        = true
    n_jobs::Option{Int} = nothing
end
MMI.fitted_params(model::LinearRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )
add_human_name_trait(LinearRegressor, "ordinary least-squares regressor (OLS)")

# =============================================================================
const OrthogonalMatchingPursuitRegressor_ = sklm(:OrthogonalMatchingPursuit)
@sk_reg mutable struct OrthogonalMatchingPursuitRegressor <: MMI.Deterministic
    n_nonzero_coefs::Option{Int} = nothing
    tol::Option{Float64} = nothing
    fit_intercept::Bool  = true
    normalize::Bool      = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
end
MMI.fitted_params(model::OrthogonalMatchingPursuitRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# =============================================================================
const OrthogonalMatchingPursuitCVRegressor_ = sklm(:OrthogonalMatchingPursuitCV)
@sk_reg mutable struct OrthogonalMatchingPursuitCVRegressor <: MMI.Deterministic
    copy::Bool            = true
    fit_intercept::Bool   = true
    # TODO Remove this when python ScikitLearn releases v1.4
    normalize::Bool       = false
    max_iter::Option{Int} = nothing::(_ === nothing||_ > 0)
    cv::Any               = 5
    n_jobs::Option{Int}   = 1
    verbose::Union{Bool,Int} = false
end
MMI.fitted_params(model::OrthogonalMatchingPursuitCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    n_nonzero_coefs = fitresult.n_nonzero_coefs_
    )
add_human_name_trait(OrthogonalMatchingPursuitCVRegressor, "orthogonal ,atching pursuit "*
                     "(OMP) model $CV")

# =============================================================================
const PassiveAggressiveRegressor_ = sklm(:PassiveAggressiveRegressor)
@sk_reg mutable struct PassiveAggressiveRegressor <: MMI.Deterministic
    C::Float64                   = 1.0::(_ > 0)
    fit_intercept::Bool          = true
    max_iter::Int                = 1_000::(_ > 0)
    tol::Float64                 = 1e-4::(_ > 0)
    early_stopping::Bool         = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int        = 5::(_ > 0)
    shuffle::Bool                = true
    verbose::Union{Bool,Int}     = 0
    loss::String                 = "epsilon_insensitive"::(_ in ("epsilon_insensitive","squared_epsilon_insensitive"))
    epsilon::Float64             = 0.1::(_ > 0)
    random_state::Any            = nothing
    warm_start::Bool             = false
    average::Union{Bool,Int}     = false
end
MMI.fitted_params(model::PassiveAggressiveRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# =============================================================================
const RANSACRegressor_ = sklm(:RANSACRegressor)
@sk_reg mutable struct RANSACRegressor <: MMI.Deterministic
    estimator::Any         = nothing
    min_samples::Union{Int,Float64}     = 5::(_ isa Int ? _ ≥ 1 : (0 ≤ _ ≤ 1))
    residual_threshold::Option{Float64} = nothing
    is_data_valid::Any          = nothing
    is_model_valid::Any         = nothing
    max_trials::Int             = 100::(_ > 0)
    max_skips::Int              = typemax(Int)::(_ > 0)
    stop_n_inliers::Int         = typemax(Int)::(_ > 0)
    stop_score::Float64         = Inf::(_ > 0)
    stop_probability::Float64   = 0.99::(0 ≤ _ ≤ 1.0)
    loss::Union{Function,String}= "absolute_error"::((_ isa Function) || _ in ("absolute_error","squared_error"))
    random_state::Any           = nothing
end
MMI.fitted_params(m::RANSACRegressor, (f, _, _)) = (
    estimator             = f.estimator_,
    n_trials              = f.n_trials_,
    inlier_mask           = f.inlier_mask_,
    n_skips_no_inliers    = f.n_skips_no_inliers_,
    n_skips_invalid_data  = f.n_skips_invalid_data_,
    n_skips_invalid_model = f.n_skips_invalid_model_
    )

# =============================================================================
const RidgeRegressor_ = sklm(:Ridge)
@sk_reg mutable struct RidgeRegressor <: MMI.Deterministic
    alpha::Union{Float64,Vector{Float64}} = 1.0::(all(_ .> 0))
    fit_intercept::Bool = true
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    solver::String      = "auto"::(_ in ("auto","svd","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any   = nothing
end
MMI.fitted_params(model::RidgeRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# =============================================================================
const RidgeCVRegressor_ = sklm(:RidgeCV)
@sk_reg mutable struct RidgeCVRegressor <: MMI.Deterministic
    alphas::Any              = (0.1, 1.0, 10.0)::(all(_ .> 0))
    fit_intercept::Bool      = true
    scoring::Any             = nothing
    cv::Any                  = 5
    gcv_mode::Option{String} = nothing::(_ === nothing || _ in ("auto","svd","eigen"))
    store_cv_values::Bool    = false
end
MMI.fitted_params(model::RidgeCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    cv_values = model.store_cv_values ? fitresult.cv_values_ : nothing
    )
add_human_name_trait(RidgeCVRegressor, "ridge regressor $CV")

# =============================================================================
const SGDRegressor_ = sklm(:SGDRegressor)
@sk_reg mutable struct SGDRegressor <: MMI.Deterministic
    loss::String             = "squared_error"::(_ in ("squared_error","huber","epsilon_insensitive","squared_epsilon_insensitive"))
    penalty::String          = "l2"::(_ in ("none","l2","l1","elasticnet"))
    alpha::Float64           = 1e-4::(_ > 0)
    l1_ratio::Float64        = 0.15::(_ > 0)
    fit_intercept::Bool      = true
    max_iter::Int            = 1_000::(_ > 0)
    tol::Float64             = 1e-3::(_ > 0)
    shuffle::Bool            = true
    verbose::Union{Int,Bool} = 0
    epsilon::Float64         = 0.1
    random_state::Any        = nothing
    learning_rate::String    = "invscaling"::(_ in ("constant","optimal","invscaling","adaptive"))
    eta0::Float64            = 0.01::(_ > 0)
    power_t::Float64         = 0.25::(_ > 0)
    early_stopping::Bool     = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int    = 5::(_ > 0)
    warm_start::Bool         = false
    average::Union{Int,Bool} = false
end
MMI.fitted_params(model::SGDRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    average_coef      = model.average ? fitresult.average_coef_ : nothing,
    average_intercept = model.average ? ifelse(model.fit_intercept, fitresult.average_intercept_, nothing) : nothing
    )
add_human_name_trait(SGDRegressor, "stochastic gradient descent-based regressor")

# =============================================================================
const TheilSenRegressor_ = sklm(:TheilSenRegressor)
@sk_reg mutable struct TheilSenRegressor <: MMI.Deterministic
    fit_intercept::Bool = true
    copy_X::Bool        = true
    max_subpopulation::Int    = 10_000::(_ > 0)
    n_subsamples::Option{Int} = nothing::(_ === nothing||_ > 0)
    max_iter::Int       = 300::(_ > 0)
    tol::Float64        = 1e-3::(_ > 0)
    random_state::Any   = nothing
    n_jobs::Option{Int} = nothing
    verbose::Bool       = false
end
MMI.fitted_params(model::TheilSenRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    breakdown       = fitresult.breakdown_,
    n_subpopulation = fitresult.n_subpopulation_
    )
add_human_name_trait(TheilSenRegressor, "Theil-Sen regressor")


# Metadata for Continuous -> Vector{Continuous}
const SKL_REGS_SINGLE = Union{
        Type{<:ARDRegressor},
        Type{<:BayesianRidgeRegressor},
        Type{<:ElasticNetRegressor},
        Type{<:ElasticNetCVRegressor},
        Type{<:HuberRegressor},
        Type{<:LarsRegressor},
        Type{<:LarsCVRegressor},
        Type{<:LassoRegressor},
        Type{<:LassoCVRegressor},
        Type{<:LassoLarsRegressor},
        Type{<:LassoLarsCVRegressor},
        Type{<:LassoLarsICRegressor},
        Type{<:LinearRegressor},
        Type{<:OrthogonalMatchingPursuitRegressor},
        Type{<:OrthogonalMatchingPursuitCVRegressor},
        Type{<:PassiveAggressiveRegressor},
        Type{<:RANSACRegressor},
        Type{<:RidgeRegressor},
        Type{<:RidgeCVRegressor},
        Type{<:SGDRegressor},
        Type{<:TheilSenRegressor}
    }

MMI.input_scitype(::SKL_REGS_SINGLE)  = Table(Continuous)
MMI.target_scitype(::SKL_REGS_SINGLE) = AbstractVector{Continuous}
