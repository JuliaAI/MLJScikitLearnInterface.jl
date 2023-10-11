
const ERR_TABLE_TYPE(t) = ArgumentError(
    "Error: Expected a table or matrix of appropriate element types but got a data of type $t."
)

function get_column_names(X)
    Tables.istable(X) || throw((ERR_TABLE_TYPE(typeof(X))))
    # Get the column names using Tables.columns or the first row
    # Former is efficient for column tables, latter is efficient for row tables
    if Tables.columnaccess(X)
        columns = Tables.columns(X)
        names = Tables.columnnames(columns)
    else
        iter = iterate(Tables.rows(X))
        names = iter === nothing ? () : Tables.columnnames(first(iter))
    end
    return names
end

function get_column_names(X::AbstractMatrix)
    n_cols = size(X, 2)
    return Symbol.("x" .* string.(1:n_cols))
end
