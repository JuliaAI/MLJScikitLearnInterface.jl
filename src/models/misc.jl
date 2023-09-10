const DummyRegressor_ = skdu(:DummyRegressor)
@sk_reg mutable struct DummyRegressor <: MMI.Deterministic
    strategy::String = "mean"::(_ in ("mean", "median", "quantile", "constant"))
    constant::Any     = nothing
    quantile::Float64 = 0.5::(0 ≤ _ ≤ 1)
end
MMI.fitted_params(m::DummyRegressor, (f, _, _)) = (
    constant  = pyconvert(Array, f.constant_),
    n_outputs = pyconvert(Int, f.n_outputs_)
    )
meta(DummyRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    weights = false
    )

"""
$(MMI.doc_header(DummyRegressor))

DummyRegressor is a regressor that makes predictions using simple rules.

"""
DummyRegressor

# ----------------------------------------------------------------------------
const DummyClassifier_ = skdu(:DummyClassifier)
@sk_clf mutable struct DummyClassifier <: MMI.Probabilistic
    strategy::String  = "stratified"::(_ in ("stratified", "most_frequent", "prior", "uniform", "constant"))
    constant::Any     = nothing
    random_state::Any = nothing
end
MMI.fitted_params(m::DummyClassifier, (f, _, _)) = (
    classes   = pyconvert(Array, f.classes_),
    n_classes = pyconvert(Int, f.n_classes_),
    n_outputs = pyconvert(Int, f.n_outputs_)
    )
meta(DummyClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
     )

"""
$(MMI.doc_header(DummyClassifier))

DummyClassifier is a classifier that makes predictions using simple rules.

"""
DummyClassifier

# ============================================================================
const GaussianNBClassifier_ = sknb(:GaussianNB)
@sk_clf mutable struct GaussianNBClassifier <: MMI.Probabilistic
    priors::Option{AbstractVector{Float64}} = nothing::(_ === nothing || all(_ .≥ 0))
    var_smoothing::Float64                  = 1e-9::(_ > 0)
end
MMI.fitted_params(m::GaussianNBClassifier, (f, _, _)) = (
    class_prior = pyconvert(Array, f.class_prior_),
    class_count = pyconvert(Array, f.class_count_),
    theta       = pyconvert(Array, f.theta_),
    var         = pyconvert(Array, f.var_),
    epsilon     = pyconvert(Float64, f.epsilon_),
    )
meta(GaussianNBClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "Gaussian naive Bayes classifier"
    )

# ============================================================================
const BernoulliNBClassifier_ = sknb(:BernoulliNB)
@sk_clf mutable struct BernoulliNBClassifier <: MMI.Probabilistic
    alpha::Float64            = 1.0::(_ ≥ 0)
    binarize::Option{Float64} = 0.0
    fit_prior::Bool           = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
end
MMI.fitted_params(m::BernoulliNBClassifier, (f, _, _)) = (
    class_log_prior  = pyconvert(Array, f.class_log_prior_),
    feature_log_prob = pyconvert(Array, f.feature_log_prob_),
    class_count      = pyconvert(Array, f.class_count_),
    feature_count    = pyconvert(Array, f.feature_count_)
    )
meta(BernoulliNBClassifier,
    input   = Table(Count),      # it expects binary but binarize takes care of that
    target  = AbstractVector{<:Finite},
     weights = false,
     human_name = "Bernoulli naive Bayes classifier"
     )

"""
$(MMI.doc_header(BernoulliNBClassifier))

Binomial naive bayes classifier. It is suitable for classification
with binary features; features will be binarized based on the
`binarize` keyword (unless it's `nothing` in which case the features
are assumed to be binary).

"""
BernoulliNBClassifier

# ============================================================================
const MultinomialNBClassifier_ = sknb(:MultinomialNB)
@sk_clf mutable struct MultinomialNBClassifier <: MMI.Probabilistic
    alpha::Float64  = 1.0::(_ ≥ 0)
    fit_prior::Bool = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
end
MMI.fitted_params(m::MultinomialNBClassifier, (f, _, _)) = (
    class_log_prior  = pyconvert(Array, f.class_log_prior_),
    feature_log_prob = pyconvert(Array, f.feature_log_prob_),
    class_count      = pyconvert(Array, f.class_count_),
    feature_count    = pyconvert(Array, f.feature_count_)
    )
meta(MultinomialNBClassifier,
    input   = Table(Count),        # NOTE: sklearn may also accept continuous (tf-idf)
    target  = AbstractVector{<:Finite},
     weights = false,
     human_name = "multinomial naive Bayes classifier"
     )

"""
$(MMI.doc_header(MultinomialNBClassifier))

Multinomial naive bayes classifier. It is suitable for classification
with discrete features (e.g. word counts for text classification).

"""
MultinomialNBClassifier

# ============================================================================
const ComplementNBClassifier_ = sknb(:ComplementNB)
@sk_clf mutable struct ComplementNBClassifier <: MMI.Probabilistic
    alpha::Float64  = 1.0::(_ ≥ 0)
    fit_prior::Bool = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
    norm::Bool      = false
end
MMI.fitted_params(m::ComplementNBClassifier, (f, _, _)) = (
    class_log_prior  = pyconvert(Array, f.class_log_prior_),
    feature_log_prob = pyconvert(Array, f.feature_log_prob_),
    class_count      = pyconvert(Array, f.class_count_),
    feature_count    = pyconvert(Array, f.feature_count_),
    feature_all      = pyconvert(Array, f.feature_all_)
    )
meta(ComplementNBClassifier,
    input   = Table(Count),        # NOTE: sklearn may also accept continuous (tf-idf)
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name = "Complement naive Bayes classifier"
    )

"""
$(MMI.doc_header(ComplementNBClassifier))

Similar to [`MultinomialNBClassifier`](@ref) but with more robust
assumptions. Suited for imbalanced datasets.

"""
ComplementNBClassifier

# ============================================================================
const KNeighborsRegressor_ = skne(:KNeighborsRegressor)
@sk_reg mutable struct KNeighborsRegressor <: MMI.Deterministic
    n_neighbors::Int    = 5::(_ > 0)
    weights::Union{String,Function} = "uniform"::((_ isa Function) || _ in ("uniform", "distance"))
    algorithm::String   = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int      = 30::(_ > 0)
    p::Int              = 2::(_ ≥ 0)
    metric::Any         = "minkowski"
    metric_params::Any  = nothing
    n_jobs::Option{Int} = nothing
end
MMI.fitted_params(m::KNeighborsRegressor, (f, _, _)) = (
    effective_metric        = f.effective_metric_,
    effective_metric_params = f.effective_metric_params_
    )
meta(KNeighborsRegressor,
    input=Table(Continuous),
    target=AbstractVector{Continuous},
    weights=false,
     human_name = "K-nearest neighbors regressor"
     )

# ----------------------------------------------------------------------------
const KNeighborsClassifier_ = skne(:KNeighborsClassifier)
@sk_clf mutable struct KNeighborsClassifier <: MMI.Probabilistic
    n_neighbors::Int    = 5::(_ > 0)
    weights::Union{String,Function} = "uniform"::((_ isa Function) || _ in ("uniform", "distance"))
    algorithm::String   = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int      = 30::(_ > 0)
    p::Int              = 2::(_ ≥ 0)
    metric::Any         = "minkowski"
    metric_params::Any  = nothing
    n_jobs::Option{Int} = nothing
end
MMI.fitted_params(m::KNeighborsClassifier, (f, _, _)) = (
    classes                 = pyconvert(Array, f.classes_),
    effective_metric        = f.effective_metric_,
    effective_metric_params = f.effective_metric_params_,
    outputs_2d              = pyconvert(Bool, f.outputs_2d_)
    )
meta(KNeighborsClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    human_name   = "K-nearest neighbors classifier"
    )
