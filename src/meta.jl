function descr(T, d)
    ex = quote
        MMI.docstring(::Type{<:$T}) = MMI.docstring_ext($T; descr=$d)
    end
    eval(ex)
    return
end

function meta(T; input=Unknown, target=Unknown, output=Unknown,
              weights::Bool=false, descr::String="")
    ex = quote
        MMI.input_scitype(::Type{<:$T})    = $input
        MMI.output_scitype(::Type{<:$T})   = $output
        MMI.target_scitype(::Type{<:$T})   = $target
        MMI.supports_weights(::Type{<:$T}) = $weights
        MMI.docstring(::Type{<:$T}) = MMI.docstring_ext($T; descr=$descr)
    end
    eval(ex)
    return
end
