"""
    macro sk_reg(expr)

Helper macro for defining interfaces of ScikitLearn regression models.
Struct fields require a type annotation and a default value as in the example
below. Constraints for parameters (fields) are introduced as for `field3`.
The constraint must refer to the parameter as `arg` or `_`. If the used
parameter does not meet a constraint the default value will be used.

@sk_reg mutable struct SomeRegression <: MMI.Deterministic
    field1::Int = 1
    field2::Any = nothing
    field3::Float64 = 0.5::(0 < arg < 0.8)
end

MMI.fit and MMI.predict methods are also produced. See also [`@sk_clf`](@ref)
"""
macro sk_reg(expr)
    modelname, params, clean_expr, expr = _sk_constructor(expr)
    fit_expr = _skmodel_fit_reg(modelname, params)
    _sk_finalize(modelname, clean_expr, fit_expr, expr)
end

"""
    macro sk_clf(expr)

Same as [`@sk_reg`](@ref) but for classifiers.
"""
macro sk_clf(expr)
    modelname, params, clean_expr, expr = _sk_constructor(expr)
    fit_expr = _skmodel_fit_clf(modelname, params)
    _sk_finalize(modelname, clean_expr, fit_expr, expr)
end

"""
    macro sk_uns(expr)

Same as [`@sk_reg`](@ref) but for unsupervised models.
"""
macro sk_uns(expr)
    modelname, params, clean_expr, expr = _sk_constructor(expr)
    fit_expr = _skmodel_fit_uns(modelname, params)
    _sk_finalize(modelname, clean_expr, fit_expr, expr)
end

# =================
# HELPER FUNCTIONS
# =================

"""
    _sk_constructor(expr)

Extracts the relevant information from the expr and build the expression
corresponding to the model constructor (see [`_model_constructor`](@ref)).
"""
function _sk_constructor(expr)
    # similar to @mlj_model
    expr, modelname, params, defaults, constraints = _process_model_def(expr)
    # keyword constructor
    const_expr = _model_constructor(modelname, params, defaults)
    # associate the constructor with the definition of the struct
    push!(expr.args[3].args, const_expr)
    # cleaner
    clean_expr = _model_cleaner(modelname, defaults, constraints)
    # return
    return modelname, params, clean_expr, expr
end

"""
    _sk_finalize(expr)

Call the different bits and pieces of code and assemble the final expression
which will:
* generate a constructor
* generate a `clean!` method
* generate the fit/predict/transform method as appropriate
* add traits.
"""
function _sk_finalize(m, clean_expr, fit_expr, expr)
    # call a different predict based on whether the model is  probabilistic,
    # deteterministic or unsupervised
    if expr.args[2].args[2] == :(MMI.Deterministic)
        predict_expr = _skmodel_predict(m)
    elseif expr.args[2].args[2] == :(MMI.Probabilistic)
        predict_expr = _skmodel_predict_prob(m)
    else
        predict_expr = nothing
    end
    expr = quote
        export $m       # make the model available
        $expr           # effectively the constructor
        $clean_expr     # clean! method
        $fit_expr       # fit method
        $predict_expr   # predict method if relevant
        MMI.load_path(::Type{<:$m})       = "MLJScikitLearnInterface.$(MMI.name($m))"
        MMI.package_name(::Type{<:$m})    = SK_NAME
        MMI.is_pure_julia(::Type{<:$m})   = false
        MMI.package_license(::Type{<:$m}) = SK_LIC
        MMI.package_uuid(::Type{<:$m})    = SK_UUID
        MMI.package_url(::Type{<:$m})     = SK_URL
        MMI.is_wrapper(::Type{<:$m})      = true
    end
    esc(expr)
end

# =================================
# Specifics for SUPERVISED MODELS
# =================================

"""
    _skmodel_fit_reg(modelname, params)

Called as part of [`@sk_reg`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn regression model.
"""
function _skmodel_fit_reg(modelname, params)
    expr = quote
        function MMI.fit(model::$modelname, verbosity::Int, X, y)
            # set X and y into a format that can be processed by sklearn
            Xmatrix   = MMI.matrix(X)
            yplain    = y
            targnames = nothing
            # check if it's a multi-target regression case, in that case keep
            # track of the names of the target columns so that the prediction
            # can be named accordingly
            if MMI.istable(y)
               yplain    = MMI.matrix(y)
               targnames = MMI.schema(y).names
            end
            # --------------------------------------------------------------
            # NOTE: be very careful before modifying the next few lines,
            # in particular, consider the following recommendations:
            # JuliaPy/PyCall.jl#using-pycall-from-julia-modules
            # --------------------------------------------------------------
            # sksym is the namespace (e.g. :SKLM)
            # skmod is the submodule (e.g. :linear_model)
            # mdl is the actual model (e.g. :ARDRegression)
            sksym, skmod, mdl = $(Symbol(modelname, "_"))
            # retrieve the namespace, if it's not there yet, import it
            parent = eval(sksym)
            ispynull(parent) && ski!(parent, skmod)
            # retrieve the effective ScikitLearn constructor
            skconstr = getproperty(parent, mdl)
            # build the scikitlearn model passing all the parameters
            skmodel = skconstr(
                        $((Expr(:kw, p, :(model.$p)) for p in params)...))
            # --------------------------------------------------------------
            # fit and organise results
            fitres = SK.fit!(skmodel, Xmatrix, yplain)
            # TODO: we may want to use the report later on
            report = NamedTuple()
            # the first nothing is so that we can use the same predict for
            # regressors and classifiers
            return ((fitres, nothing, targnames), nothing, report)
        end
    end
end

"""
    _skmodel_fit_clf(modelname, params)

Called as part of [`@sk_clf`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn classifier model.
"""
function _skmodel_fit_clf(modelname, params)
    quote
        function MMI.fit(model::$modelname, verbosity::Int, X, y)
            Xmatrix = MMI.matrix(X)
            yplain  = MMI.int(y)
            # See _skmodel_fit_reg, same story
            sksym, skmod, mdl = $(Symbol(modelname, "_"))
            parent = eval(sksym)
            ispynull(parent) && ski!(parent, skmod)
            skconstr = getproperty(parent, mdl)
            skmodel = skconstr(
                        $((Expr(:kw, p, :(model.$p)) for p in params)...))
            fitres  = SK.fit!(skmodel, Xmatrix, yplain)
            # TODO: we may want to use the report later on
            report  = NamedTuple()
            # pass y[1] for decoding in predict method, first nothing
            # is targnames
            return ((fitres, y[1], nothing), nothing, report)
        end
    end
end

"""
    _skmodel_predict(modelname)

Called as part of [`@sk_model`](@ref), returns the expression corresponing to
the `predict` method for the ScikitLearn model (for a deterministic model).
"""
function _skmodel_predict(modelname)
    quote
        function MMI.predict(model::$modelname, (fitres, y1, targnames), Xnew)
            Xmatrix = MMI.matrix(Xnew)
            preds   = SK.predict(fitres, Xmatrix)
            if isa(preds, Matrix)
                # only regressors are possibly multitarget;
                # build a table with the appropriate column names
                return MMI.table(preds, names=targnames)
            end
            if y1 !== nothing
                # if it's a classifier
                return preds |> MMI.decoder(y1)
            end
            return preds
        end
    end
end

"""
    _skmodel_predict_prob(modelname)

Same as `_skmodel_predict` but with probabilities. Note that only classifiers
are probabilistic in sklearn so that we always decode.
"""
function _skmodel_predict_prob(modelname)
    quote
        # there are no multi-task classifiers in sklearn
        function MMI.predict(model::$modelname, (fitres, y1, _), Xnew)
            Xmatrix = MMI.matrix(Xnew)
            # this is an array of size n x c with rows that sum to 1
            preds   = SK.predict_proba(fitres, Xmatrix)
            classes = MMI.classes(y1)
            return [MMI.UnivariateFinite(classes, preds[i, :])
                    for i in 1:size(Xmatrix, 1)]
        end
    end
end

# ============================================================================
# Specifics for UNSUPERVISED MODELS
# ============================================================================
# Depending on the model there may be
# * a transform
# * a inverse_transform
# * a predict

"""
    _skmodel_fit_uns(modelname, params)

Called as part of [`@sk_uns`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn unsupervised model.
"""
function _skmodel_fit_uns(modelname, params)
    quote
        function MMI.fit(model::$modelname, verbosity::Int, X)
            Xmatrix = MMI.matrix(X)
            skmodel = $(Symbol(modelname, "_"))(
                        $((Expr(:kw, p, :(model.$p)) for p in params)...))
            fitres  = SK.fit!(skmodel, Xmatrix)
            # TODO: we may want to use the report later on
            report  = NamedTuple()
            return (fitres, nothing, report)
        end
    end
end

"""
    macro sku_transform(modelname)

Adds a `transform` method to a declared scikit unsupervised model if
there is one supported.
"""
macro sku_transform(modelname)
    quote
        function MMI.transform(::$modelname, fitres, X)
            X = SK.transform(fitres, MMI.matrix(X))
            MMI.table(X)
        end
    end
end

"""
    macro sku_inverse_transform(modelname)

Adds an `inverse_transform` method to a declared scikit unsupervised model if
there is one supported.
"""
macro sku_inverse_transform(modelname)
    quote
        function MMI.inverse_transform(::$modelname, fitres, X)
            X = SK.inverse_transform(fitres, MMI.matrix(X))
            MMI.table(X)
        end
    end
end

"""
    macro sku_predict

Adds a `predict` method to a declared scikit unsupervised model if there is one
supported. Returns a MMI.categorical vector.
Only `AffinityPropagation`, `Birch`, `KMeans`, `MiniBatchKMeans` and
`MeanShift` support a `predict` method.

Note: for models that offer a `fit_predict`, the encoding is done in the
`fitted_params`.
"""
macro sku_predict(modelname)
    quote
        function MMI.predict(m::$modelname, fitres, X)
            # this is due to the fact that we have nested modules
            # so potentially have to extract the leaf node...
            sm = Symbol($modelname)
            ss = string(sm)
            sm = Symbol(split(ss, ".")[end])
            if sm in (:Birch, :KMeans, :MiniBatchKMeans)
                catv = MMI.categorical(1:m.n_clusters)
            elseif sm == :AffinityPropagation
                nc   = length(fitres.cluster_centers_indices_)
                catv = MMI.categorical(1:nc)
            elseif sm == :MeanShift
                nc   = size(fitres.cluster_centers_, 1)
                catv = MMI.categorical(1:nc)
            else
                throw(ArgumentError("Model $sm does not support `predict`."))
            end
            preds  = SK.predict(fitres, MMI.matrix(X)) .+ 1
            return catv[preds]
        end
    end
end
