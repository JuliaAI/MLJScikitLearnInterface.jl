# Additional tests following bug reports

@testset "i10" begin
    X, _ = MB.make_blobs(500, 3, rng=555)
    m = AgglomerativeClustering()
    f, = MB.fit(m, 1, X)
    @test f !== nothing
end
