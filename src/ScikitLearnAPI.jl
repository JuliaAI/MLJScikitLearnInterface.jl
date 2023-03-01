# The mapping of Julia functions into python functions in this file is adapted from 
# code in ScikitLearn.jl, written by Cédric St-Jean. 
module ScikitLearnAPI

using PythonCall

const numpy = PythonCall.pynew()
const sklearn = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(numpy, pyimport("numpy"))
    PythonCall.pycopy!(sklearn, pyimport("sklearn"))
end

# convert return values back to Julia
tweak_rval(x) = x
function tweak_rval(x::Py)
    if pyisinstance(x, numpy.ndarray)
        return pyconvert(Array, x)
    else
        return pyconvert(Any, x)
    end
end

################################################################################
# Julia => Python
################################################################################
api_map = Dict(:decision_function => :decision_function,
               :fit_predict! => :fit_predict,
               :fit_transform! => :fit_transform,
               :get_feature_names => :get_feature_names,
               :get_params => :get_params,
               :predict => :predict,
               :predict_proba => :predict_proba,
               :predict_log_proba => :predict_log_proba,
               :partial_fit! => :partial_fit,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform,
               :inverse_transform => :inverse_transform,
               :set_params! => :set_params)
               
for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_estimator::Py, args...; kwargs...) =
        tweak_rval(py_estimator.$(py_fun)(args...; kwargs...))
end

fit!(py_estimator::Py, args...; kwargs...) = py_estimator.fit(args...; kwargs...)

end
