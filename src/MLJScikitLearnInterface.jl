module MLJScikitLearnInterface 

import MLJModelInterface
import MLJModelInterface:
        @mlj_model, _process_model_def, _model_constructor, _model_cleaner,
        Table, Continuous, Count, Finite, OrderedFactor, Multiclass, Unknown
const MMI = MLJModelInterface

import ScikitLearn
function __init__()
    ScikitLearn.Skcore.import_sklearn()
end
const SK = ScikitLearn

# Note: PyCall is already imported as part of ScikitLearn so this is cheap
import PyCall: ispynull, PyNULL, pyimport

# ------------------------------------------------------------------------
# NOTE: the next few lines of wizardry and their call should not be
# modified carelessly. In particular do consider the recommendations at
# JuliaPy/PyCall.jl#using-pycall-from-julia-modules
# from which much of this stems.

# supervised
const SKLM = PyNULL()
const SKGP = PyNULL()
const SKEN = PyNULL()
const SKDU = PyNULL()
const SKNB = PyNULL()
const SKNE = PyNULL()
const SKDA = PyNULL()
const SKSV = PyNULL()
sklm(m) = (:SKLM, :linear_model, m)
skgp(m) = (:SKGP, :gaussian_process, m)
sken(m) = (:SKEN, :ensemble, m)
skdu(m) = (:SKDU, :dummy, m)
sknb(m) = (:SKNB, :naive_bayes, m)
skne(m) = (:SKNE, :neighbors, m)
skda(m) = (:SKDA, :discriminant_analysis, m)
sksv(m) = (:SKSV, :svm, m)

# unsupervised
const SKCL = PyNULL()
skcl(m) = (:SKCL, :cluster, m)

# Generic loader (see _skmodel_fit_* in macros)
ski!(sks, mdl) = copy!(sks, pyimport("sklearn.$mdl"))
# ------------------------------------------------------------------------

const Option{T} = Union{Nothing,T}

# recurrent information for traits
const PKG_NAME = "ScikitLearn"
const API_PKG_NAME = "MLJScikitLearnInterface"
const SK_UUID = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
const SK_URL  = "https://github.com/cstjean/ScikitLearn.jl"
const SK_LIC  = "BSD"

const CV = "with built-in cross-validation"

include("macros.jl")
include("meta.jl")

include("models/linear-regressors.jl")
include("models/linear-regressors-multi.jl")
include("models/linear-classifiers.jl")
include("models/gaussian-process.jl")
include("models/ensemble.jl")
include("models/discriminant-analysis.jl")
include("models/svm.jl")
include("models/misc.jl")

include("models/clustering.jl")

end
