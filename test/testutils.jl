SKD = pyimport("sklearn.datasets")

rmse(y, ŷ) = sqrt(sum(abs2, ŷ .- y) / length(y))

function simple_regression(n=100, p=3; sigma=0.1, σ=sigma, seed=6168901234)
    r = StableRNG(seed)
    X = hcat(randn(r, n, p), ones(n))
    θ = randn(r, p + 1)
    y = X * θ + σ * randn(r, n)
    ŷ = X * (X \ y)
    return X, y, rmse(y, ŷ)
end

function test_regression(m, X, y, ls)
    f, _, _ = MB.fit(m, 1, X, y)
    ŷ = MB.predict(m, f, X)
    return rmse(y, ŷ) / ls, f
end

function simple_regression_2(n=100, p=3;
                             sigma=0.1, σ=sigma, seed=6168901234)
    r  = StableRNG(seed)
    X  = hcat(randn(r, n, p), ones(n))
    θ  = randn(r, p + 1)
    y1 = X * θ + σ * randn(r, n)
    y2 = X * θ + σ * randn(r, n)
    ŷ  = X * (vcat(X, X) \ vcat(y1, y2))
    return X, hcat(y1, y2), (rmse(y1, ŷ)+rmse(y2, ŷ))/2
end

function test_regression_2(m, X, y, ls)
    f, _, _ = MB.fit(m, 1, X, y)
    ŷ = MB.predict(m, f, X)
    return sqrt(sum(abs2, vec(MB.matrix(y) .- MB.matrix(ŷ)))) / ls, f
end

function simple_binaryclf(n=100, p=3; sigma=0.1, seed=616866614)
    X, y = SKD.make_blobs(n_samples=n, n_features=p, centers=2, random_state=seed)
    return MB.table(pyconvert(Array, X)), MB.categorical(pyconvert(Array, y))
end

function test_clf(m, X, y)
    f, _, _ = MB.fit(m, 1, X, y)
    if typeof(m) <: MB.Deterministic
        ŷ = MB.predict(m, f, X)
    else
        ŷ = MB.predict_mode(m, f, X)
    end
    return MB.accuracy(ŷ, y), f
end
