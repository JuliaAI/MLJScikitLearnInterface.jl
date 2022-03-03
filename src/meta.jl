function add_human_name_trait(T, d)
    ex = quote
        MMI.human_name(::Type{<:$T}) = $d
    end
    eval(ex)
    return
end

function meta(T; input=Unknown, target=Unknown, output=Unknown,
              weights::Bool=false, human_name::String="")
    ex = quote
        MMI.input_scitype(::Type{<:$T})    = $input
        MMI.output_scitype(::Type{<:$T})   = $output
        MMI.target_scitype(::Type{<:$T})   = $target
        MMI.supports_weights(::Type{<:$T}) = $weights
        isempty($human_name) || (MMI.human_name(::Type{<:$T}) = $human_name)
    end
    eval(ex)
    return
end
