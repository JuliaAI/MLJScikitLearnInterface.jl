@testset "reg-mdl" begin
    m = ARDRegressor()
    @test m isa MB.Deterministic
    @test m.max_iter > 0
    # set
    m = ARDRegressor(max_iter=100)
    @test m.max_iter == 100
    # clean method
    @test @test_logs (:warn, r"Constraint") ARDRegressor(max_iter=-5).max_iter > 0

    # traits
    @test MB.load_path(m) == "MLJScikitLearnInterface.ARDRegressor"
    @test MB.package_name(m) == "MLJScikitLearnInterface"
end

@testset "clf-mdl" begin
    m = LogisticClassifier()
    @test m isa MB.Probabilistic
    @test m.penalty == "l2"
    # set
    m = LogisticClassifier(fit_intercept=false)
    @test !m.fit_intercept
    # clean
    @test @test_logs (:warn, r"Constraint") LogisticClassifier(
            penalty="foo").penalty == "l2"

    # traits
    @test MB.load_path(m) == "MLJScikitLearnInterface.LogisticClassifier"
    @test MB.package_license(m) == "BSD"
end
