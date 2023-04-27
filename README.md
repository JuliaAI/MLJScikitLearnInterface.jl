# MLJ <> scikit-learn
Repository implementing MLJ interface for 
[scikit-learn](https://github.com/scikit-learn/scikit-learn) models (via PythonCall.jl). 


[![Build Status](https://github.com/JuliaAI/MLJScikitLearnInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJScikitLearnInterface.jl/actions)
[![Coverage](http://codecov.io/github/JuliaAI/MLJScikitLearnInterface.jl/coverage.svg?branch=master)](https://codecov.io/gh/JuliaAI/MLJScikitLearnInterface.jl)

# Known Issue
If you are using Linux and Julia version `>=1.8.3`, the `libstdcxx` version is not compatible with `scikit-learn>=1.2`. To get around this issue, you have to swap out the `libstdcxx` version that is loaded in with Julia. There is two methods to do this. The first is to build an environment with `Conda.jl` and use that as your `LD_LIBRARY_PATH`. 
```bash
ROOT_ENV=`julia -e "using Conda; print(Conda.ROOTENV)`
export LD_LIBRARY_PATH=$ROOT_ENV"/lib":$LD_LIBRARY_PATH
```

Another method is to link your OS's version of `libstdcxx` instead of the one that comes with Julia. More details can be found [here](https://github.com/hhaensel/ReplaceLibstdcxx.jl).

In both cases, it is recommended to use a Julia version `>=1.8.4`, but these fixes can be used as a last resort if you have to stay on an older Julia version. 

# Related Projects
- [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) - a Julia implementation of the scikit-learn API
