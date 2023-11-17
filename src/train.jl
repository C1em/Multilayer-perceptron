import ArgParse, ProfileView

include("mlp.jl")
using .MLP
using Profile

Optional{T} = Union{Missing,Nothing,T}

function parse_commandline(ARGS)
    s = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table s begin
        "--dataset"
        help = "the dataset in csv format"
        required = true
        "--dataset-valid"
        help = "the validation dataset in csv format"
        required = false
    end

    return ArgParse.parse_args(ARGS, s)
end

function read_dataset_lines(dataset_file)
    dataset_lines = []
    try
        dataset_lines = readlines(dataset_file)
    catch e
        println("bad formatted dataset")
        throw(e)
    end
    return dataset_lines
end

function convert_dataset_to_matrix(dataset_lines)::Tuple{Matrix,Matrix}
    matrix = Matrix{Optional{Float64}}(nothing, 0, 31)
    results = []
    # converter = Dict("M" => 1, "B" => 1)
    converter = Dict("M" => [1, 0], "B" => [0, 1])

    dataset_lines = split.(dataset_lines, ",")
    results = splice!.(dataset_lines, 2)
    results = [converter[result] for result in results]
    results = reduce(hcat, results)'
    dataset_lines = [parse.(Float64, dataset_line) for dataset_line in dataset_lines]
    matrix = reduce(hcat, dataset_lines)'

    means = sum.(eachcol(matrix)) / size(matrix, 1)
    stds = []
    for i = eachindex(means)
        push!(
            stds,
            sqrt(sum((matrix[:, i] .- means[i]) .^ 2) / size(matrix, 1))
        )
    end

    for i in axes(matrix, 1)
        matrix[i, :] = (matrix[i, :] - means) ./ stds
    end

    return matrix, results
end

function main(ARGS)
    parse_args = parse_commandline(ARGS)

    dataset_lines = read_dataset_lines(parse_args["dataset"])
    X, Y = convert_dataset_to_matrix(dataset_lines)

    X_valid::Matrix{Float64} = [;;]
    Y_valid::Matrix{Int64} = [;;]
    if !isnothing(parse_args["dataset-valid"])
        dataset_valid_lines = read_dataset_lines(parse_args["dataset-valid"])
        X_valid, Y_valid = convert_dataset_to_matrix(dataset_valid_lines)
    end

    create_network([
        LayerParameters(31, "relu", "glorot_normal"),
        LayerParameters(31, "relu", "glorot_normal"),
        LayerParameters(15, "relu", "glorot_normal"),
        LayerParameters(2, "softmax", "glorot_normal"),
    ])
    train(X, Y, X_valid, Y_valid, "binary_crossentropy", 0.0001, 256, 1000)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
