module MLJScikitLearnInterface 

import MLJModelInterface
import MLJModelInterface:
        @mlj_model, _process_model_def, _model_constructor, _model_cleaner,
        Table, Continuous, Count, Finite, OrderedFactor, Multiclass, Unknown
const MMI = MLJModelInterface

include("ScikitLearnAPI.jl")
const SK = ScikitLearnAPI
import PythonCall: pyisnull, PyNULL, pyimport, pycopy!, pynew, pyconvert

# supervised
const SKLM = pynew()
const SKGP = pynew()
const SKEN = pynew()
const SKDU = pynew()
const SKNB = pynew()
const SKNE = pynew()
const SKDA = pynew()
const SKSV = pynew()
sklm(m) = (:SKLM, :linear_model, m)
skgp(m) = (:SKGP, :gaussian_process, m)
sken(m) = (:SKEN, :ensemble, m)
skdu(m) = (:SKDU, :dummy, m)
sknb(m) = (:SKNB, :naive_bayes, m)
skne(m) = (:SKNE, :neighbors, m)
skda(m) = (:SKDA, :discriminant_analysis, m)
sksv(m) = (:SKSV, :svm, m)

# unsupervised
const SKCL = pynew()
skcl(m) = (:SKCL, :cluster, m)

# Generic loader (see _skmodel_fit_* in macros)
ski!(sks, mdl) = pycopy!(sks, pyimport("sklearn.$mdl"))
# ------------------------------------------------------------------------

const Option{T} = Union{Nothing,T}

# recurrent information for traits
const PKG_NAME = "MLJScikitLearnInterface"
const API_PKG_NAME = "MLJScikitLearnInterface"
const SK_UUID = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
const SK_URL  = "https://github.com/JuliaAI/MLJScikitLearnInterface.jl"
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
