const MultiTaskLassoRegressor_ = sklm(:MultiTaskLasso)
@sk_reg mutable struct MultiTaskLassoRegressor <: MMI.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::MultiTaskLassoRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )
add_human_name_trait(MultiTaskLassoRegressor, "multi-target lasso regressor")

# ==============================================================================
const MultiTaskLassoCVRegressor_ = sklm(:MultiTaskLassoCV)
@sk_reg mutable struct MultiTaskLassoCVRegressor <: MMI.Deterministic
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 300::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    cv::Any             = 5
    verbose::Union{Bool, Int} = false
    n_jobs::Option{Int} = 1
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::MultiTaskLassoCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    mse_path  = fitresult.mse_path_,
    alphas    = fitresult.alphas_
    )
add_human_name_trait(MultiTaskLassoCVRegressor, "multi-target lasso regressor $CV")

# ==============================================================================
const MultiTaskElasticNetRegressor_ = sklm(:MultiTaskElasticNet)
@sk_reg mutable struct MultiTaskElasticNetRegressor <: MMI.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5::(0 ≤ _ ≤ 1)
    fit_intercept::Bool = true
    normalize::Bool     = true
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::MultiTaskElasticNetRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )
add_human_name_trait(MultiTaskElasticNetRegressor, "multi-target elastic net regressor")

# ==============================================================================
const MultiTaskElasticNetCVRegressor_ = sklm(:MultiTaskElasticNetCV)
@sk_reg mutable struct MultiTaskElasticNetCVRegressor <: MMI.Deterministic
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5::(0 ≤ _ ≤ 1)
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    cv::Any             = 5
    copy_X::Bool        = true
    verbose::Union{Bool,Int} = 0
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MMI.fitted_params(model::MultiTaskElasticNetCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    mse_path  = fitresult.mse_path_,
    l1_ratio  = fitresult.l1_ratio_
    )
add_human_name_trait(MultiTaskElasticNetCVRegressor, "multi-target elastic "*
                     "net regressor $CV")


const SKL_REGS_MULTI = Union{
       Type{<:MultiTaskLassoRegressor},
       Type{<:MultiTaskLassoCVRegressor},
       Type{<:MultiTaskElasticNetRegressor},
       Type{<:MultiTaskElasticNetCVRegressor}
    }

MMI.input_scitype(::SKL_REGS_MULTI)  = Table(Continuous)
MMI.target_scitype(::SKL_REGS_MULTI) = Table(Continuous)
