module MLP
export train, LayerParameters, create_network

import Distributions, Plots
ENV["GKSwstype"] = "nul"

Optional{T} = Union{Missing,Nothing,T}

struct Loss
    fun::Function
    derivative::Function
    threshold::Optional{Float64}
    Loss(fun, derivative, threshold=nothing) = new(fun, derivative, threshold)
end

struct LayerActivation
    fun::Function
    derivative::Function
end

mutable struct NetworkLayer
    weights::Optional{Matrix{Float64}}
    biases::Optional{Vector{Float64}}
    activation::LayerActivation
end
NetworkLayer(activation::LayerActivation) = NetworkLayer(nothing, nothing, activation)

layers::Vector{NetworkLayer} = []

cache_layers::Vector{Tuple{Vector,AbstractArray}} = []

struct LayerParameters
    nb_nodes::Int64
    activation::AbstractString
    weights_initializer::Optional{AbstractString}
end
LayerParameters(nb_nodes::Int64, activation::AbstractString) = LayerParameters(nb_nodes, activation, nothing)

# ACTIVATION FUNCTIONS ------------------------------------------------

# TODO: softmax
softmax(Z) = begin
    s = sum(2 .^ Z)
    return (2 .^ Z) ./ s
end

softmax_derivative(Z) = begin
    sZ = softmax(Z)
    jacobian = sZ * -sZ'
    for i in eachindex(sZ)
        jacobian[i, i] = sZ[i] * (1 - sZ[i])
    end
    return jacobian
end

sigmoid(Z) = 1 ./ (1 .+ exp2.(-Z))
sigmoid_derivative(Z) = sigmoid(Z) .* (1 .- sigmoid(Z))

relu(Z) = max.(0, Z)
relu_derivative(Z) = [z > 0 ? 1 : 0 for z in Z]

leaky_relu(Z) = [z > 0 ? z : -0.01z for z in Z]
leaky_relu_derivative(Z) = [z > 0 ? 1 : -0.01 for z in Z]

tanh(Z) = (exp2.(Z) - exp2.(-Z)) ./ (exp2.(Z) + exp2.(-Z))
tanh_derivative(Z) = 1 .- tanh.(Z) .^ 2

function find_activation_function(str::AbstractString)
    activations = Dict(
        [
        "sigmoid" => LayerActivation(sigmoid, sigmoid_derivative),
        "relu" => LayerActivation(relu, relu_derivative),
        "leaky relu" => LayerActivation(leaky_relu, leaky_relu_derivative),
        "tanh" => LayerActivation(tanh, tanh_derivative),
        "softmax" => LayerActivation(softmax, softmax_derivative)
    ]
    )
    return activations[str]
end

# LOSS FUNCTIONS --------------------------------------------------

mean_square(Y_pred, Y) = sum((Y_pred - Y) .^ 2) / length(Y_pred)
mean_square_derivative(Y_pred, Y) = (Y_pred - Y) / length(Y_pred)

binary_crossentropy(Y_pred, Y) = begin
    binary_crossentropy_one_var(y_pred, y) = y == 1 ? log2(y_pred) : log2(1 - y_pred)

    return -sum(binary_crossentropy_one_var.(Y_pred, Y))
end

binary_crossentropy_derivative(Y_pred, Y) = begin
    binary_crossentropy_derivative_one_var(y_pred, y) = y == 1 ? -1 / max(y_pred, 1e-300) : 1 / 1 - min(y_pred, 1 - 1e-300)

    return binary_crossentropy_derivative_one_var.(Y_pred, Y)
end

crossentropy(Y_pred, Y) = -sum(Y .* log2.(Y_pred))
crossentropy_derivative(Y_pred, Y) = -Y ./ Y_pred

function find_loss_function(str::AbstractString)
    losses = Dict(
        [
        "mean_square" => Loss(mean_square, mean_square_derivative),
        "binary_crossentropy" => Loss(binary_crossentropy, binary_crossentropy_derivative, 1.0 * length(layers[end].biases)),
        "crossentropy" => Loss(crossentropy, crossentropy_derivative, 1.0),
    ]
    )
    return losses[str]
end

# WEIGHTS INITIALIZER --------------------------------------------------

function glorot_uniform(n_in, n_out)
    glorot_u = Distributions.Uniform(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
    return rand(glorot_u, n_out, n_in)
end

function glorot_normal(n_in, n_out)
    glorot_n = Distributions.Normal(0, sqrt(2 / (n_in + n_out)))
    return rand(glorot_n, n_out, n_in)
end

function he_uniform(n_in, n_out)
    he_u = Distributions.Uniform(-sqrt(6 / n_in), sqrt(6 / n_in))
    return rand(he_u, n_out, n_in)
end

function he_normal(n_in, n_out)
    he_n = Distributions.Normal(0, sqrt(2 / n_in))
    return rand(he_n, n_out, n_in)
end

function lecun_uniform(n_in, n_out)
    lecun_u = Distributions.Uniform(-sqrt(1 / n_in), sqrt(1 / n_in))
    return rand(lecun_u, n_out, n_in)
end

function lecun_normal(n_in, n_out)
    lecun_n = Distributions.Normal(0, 1 / n_in)
    return rand(lecun_n, n_out, n_in)
end

function weights_initialize(initializer, n_in, n_out)
    map_name_to_fun = Dict([
        "glorot_uniform" => glorot_uniform,
        "glorot_normal" => glorot_normal,
        "he_uniform" => he_uniform,
        "he_normal" => he_normal,
        "lecun_uniform" => lecun_uniform,
        "lecun_normal" => lecun_normal
    ])
    weights_initializer_function = map_name_to_fun[initializer]
    return weights_initializer_function(n_in, n_out)
end

function create_network(layers_parameters::AbstractArray{LayerParameters})

    if length(layers_parameters) < 2
        throw(ArgumentError("create_network should take at least 2 layers"))
    end

    global layers

    i = 2
    while i <= length(layers_parameters)
        push!(
            layers,
            NetworkLayer(
                weights_initialize(
                    layers_parameters[i].weights_initializer,
                    layers_parameters[i-1].nb_nodes,
                    layers_parameters[i].nb_nodes
                ),
                zeros(Float64, layers_parameters[i].nb_nodes),
                find_activation_function(layers_parameters[i].activation)
            )
        )
        i += 1
    end
end

function predict(A)
    global layers
    for layer in layers
        #         wL-1,1 wL-1,2
        # W = wL,1
        #     wL,2
        W = layer.weights

        # B = b1 b2
        B = layer.biases

        # Z = z1 z2
        Z = W * A + B

        global cache_layers
        push!(
            cache_layers,
            (A, layer.activation.derivative(Z))
        )

        A = layer.activation.fun(Z)
    end
    return A
end

function backpropagation(layer_idx, ∂L∂𝝈, learning_rate)
    global layers
    if layer_idx == length(layers)
        Aprev, d𝝈dZ = pop!(cache_layers)

        if length(size(d𝝈dZ)) == 1
            ∂L∂Z = ∂L∂𝝈 .* d𝝈dZ
        elseif length(size(d𝝈dZ)) == 2
            ∂L∂Z = d𝝈dZ' * ∂L∂𝝈
        else
            Throw(ErrorException("Error during backpropagation"))
        end

        layers[layer_idx].weights -= learning_rate * ∂L∂Z * Aprev'
        layers[layer_idx].biases -= learning_rate * ∂L∂Z

        return ∂L∂Z
    end

    Wnext = layers[layer_idx+1].weights
    ∂L∂Znext = backpropagation(layer_idx + 1, ∂L∂𝝈, learning_rate)

    global cache_layers

    Aprev, d𝝈dZ = pop!(cache_layers)

    if length(size(d𝝈dZ)) == 1
        ∂L∂Z = Wnext' * ∂L∂Znext .* d𝝈dZ
    elseif length(size(d𝝈dZ)) == 2
        ∂L∂Z = d𝝈dZ' * (Wnext' * ∂L∂Znext)
    else
        Throw(ErrorException("Error during backpropagation"))
    end

    layers[layer_idx].weights -= learning_rate * ∂L∂Z * Aprev'
    layers[layer_idx].biases -= learning_rate * ∂L∂Z

    return ∂L∂Z
end

function train(dataset, Y, dataset_valid, Y_valid, loss_fun, learning_rate, batch_size, epochs)
    loss = find_loss_function(loss_fun)
    total_losses = []
    total_valid_losses = []
    total_accuracy = []
    total_valid_accuracy = []
    predictions = []

    for _ in 1:epochs
        # X = f1 f2
        losses = []
        for (X, Y_actual) in zip(eachrow(dataset), eachrow(Y))

            Y_pred = predict(X)
            push!(predictions, Y_pred)

            actual_loss = loss.fun(Y_pred, Y_actual)
            push!(losses, actual_loss)

            loss_derivative = loss.derivative(Y_pred, Y_actual)

            backpropagation(1, loss_derivative, learning_rate)

        end
        push!(
            total_losses,
            sum(losses) / length(losses)
        )
        if !isnothing(loss.threshold)
            push!(
                total_accuracy,
                count(<(loss.threshold), losses) / length(losses)
            )
        end

        if length(dataset_valid) > 0
            Y_valid_preds = predict.(eachrow(dataset_valid))
            valid_losses = loss.fun.(Y_valid_preds, eachrow(Y_valid))
            push!(
                total_valid_losses,
                sum(valid_losses) / length(valid_losses)
            )
            if !isnothing(loss.threshold)
                push!(
                    total_valid_accuracy,
                    count(<(loss.threshold), valid_losses) / length(valid_losses)
                )
            end
        end
    end
    Plots.savefig(Plots.plot([total_losses, total_valid_losses], label=["training loss" "validation loss"]), "loss_plot.png")
    Plots.savefig(Plots.plot([total_accuracy, total_valid_accuracy], label=["training accuracy" "validation accuracy"]), "accuracy_plot.png")
    # TODO: for the batch add them to cache the divide by batch_size to get the mean then apply (also cache the loss_derivative in another cache)
end
end
