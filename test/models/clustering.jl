X, _ = MB.make_blobs(500, 3, rng=555)

fparams = (
        AffinityPropagation = (:cluster_centers_indices, :cluster_centers, :labels, :affinity_matrix),
        AgglomerativeClustering = (:n_clusters, :labels, :n_leaves, :n_connected_components, :children),
        Birch = (:root, :dummy_leaf, :subcluster_centers, :subcluster_labels, :labels),
        DBSCAN = (:core_sample_indices,  :components, :labels),
        FeatureAgglomeration = (:n_clusters, :labels, :n_leaves, :n_connected_components,  :children, :distances),
        KMeans = (:cluster_centers, :labels, :inertia),
        MiniBatchKMeans = (:cluster_centers, :labels, :inertia),
        MeanShift = (:cluster_centers, :labels),
        OPTICS = (:labels, :reachability, :ordering, :core_distances, :predecessor, :cluster_hierarchy),
        SpectralClustering = (:labels, :affinity_matrix)
    )


models = (
        AffinityPropagation,
        AgglomerativeClustering,
        Birch, 
        DBSCAN,
        MeanShift,
    )

@testset "Fit/predict" begin
   for mod in models
        m = mod()
        f, = MB.fit(m, 1, X)
        fp = MB.fitted_params(m, f)

        if m in (AffinityPropagation, MeanShift)
            p = MB.predict(m, f, X)
            @test p isa MB.CategoricalArray
        end

        @test keys(fp) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end

@testset "Scitypes" begin
    for mod in models
        m = mod()
        @test input_scitype(mod) == MB.Table(MB.Continuous)
        m == AffinityPropagation &&
            @test target_scitype(mod) == AbstractVector{MB.Multiclass}
        m == MeanShift &&
            @test output_scitype(mod) == MB.Table(MB.Continuous)
    end
end




models = (
           KMeans, MiniBatchKMeans
        )

@testset "Fit/predict/transform" begin
   for mod in models
       m = mod(n_clusters=4)
       f, = MB.fit(m, 1, X)
       fp = MB.fitted_params(m, f)
       @test size(fp.cluster_centers) == (4, 3)
       p = MB.predict(m, f, X)
       @test p isa MB.CategoricalArray
       Xt = MB.transform(m, f, X)
       @test Xt isa MB.Tables.MatrixTable

       @test keys(fp) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end

@testset "Scitypes" begin
    for mod in models
        m = mod()
        @test input_scitype(mod) == MB.Table(MB.Continuous)
        @test target_scitype(mod) == AbstractVector{MB.Multiclass}
        @test output_scitype(mod) == MB.Table(MB.Continuous)
    end
end




models = (
           Birch, FeatureAgglomeration
        )

@testset "Fit/predict/transform" begin
   for mod in models
       m = mod()
       f, = MB.fit(m, 1, X)
       fp = MB.fitted_params(m, f)
       @test fp.labels isa MB.CategoricalArray

       Xt = MB.transform(m, f, X)
       if mod == FeatureAgglomeration
           @test fp.distances === nothing
           Xit = MB.inverse_transform(m, f, Xt) # Need MB.inverse_transform function
          # NOTE: they're not equal (not sure why)
           @test Xit isa MB.Tables.MatrixTable
       else
            p = MB.predict(m, f, X)
            @test p isa MB.CategoricalArray
            @test Xt isa MB.Tables.MatrixTable
       end

       @test keys(fp) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end

@testset "Scitypes" begin
    for mod in models
        @test input_scitype(mod) == MB.Table(MB.Continuous)
        mod == Birch &&
            @test target_scitype(mod) == AbstractVector{MB.Multiclass}
        @test output_scitype(mod) == MB.Table(MB.Continuous)
    end
end



model = (
        OPTICS, SpectralClustering
        )

@testset "Fit" begin
   for mod in models
       m=mod()
       f, = MB.fit(m, 1, X)
       fp = MB.fitted_params(m, f)
       @test fp.labels isa MB.CategoricalArray
       @test keys(fp) == getproperty(fparams, Symbol(mod))
    end
end

@testset "Docstrings" begin
    for mod in models
        m = mod()
        @test !isempty(MB.docstring(m))
    end
end

@testset "Scitypes" begin
    for mod in models
        m = mod()
        @test input_scitype(mod) == MB.Table(MB.Continuous)
    end
end
