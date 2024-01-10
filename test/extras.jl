# Additional tests following bug reports

@testset "i10" begin
    X, _ = MB.make_blobs(500, 3, rng=555)
    m = AgglomerativeClustering()
    f, = MB.fit(m, 1, X)
    @test f !== nothing
end

@testset "i48" begin
    X, y = MB.make_blobs(500, 3, rng=555)
    y[2:end] .= y[1]
    models = [
        PassiveAggressiveClassifier,
        PerceptronClassifier,
        SGDClassifier,
        SVMClassifier,
        SVMNuClassifier,
        BayesianQDA,
        ProbabilisticSGDClassifier,
        SVMLinearClassifier,
        LogisticCVClassifier,
        LogisticClassifier,
        GaussianProcessClassifier,
        GradientBoostingClassifier
    ]
    for mod in models
        m = mod()
        f, = MB.fit(m, 1, X, y)
        @test f !== nothing
    end
end

@testset "i63" begin
    X, y = MB.make_blobs(500, 3, rng=555)
    w = Dict(1=>0.2, 2=>0.7, 3=>0.1)
    m = RandomForestClassifier(class_weight=w)
    f, = MB.fit(m, 1, X, y)
    @test f !== nothing
end
