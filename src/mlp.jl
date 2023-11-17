module MLP
export train, LayerParameters, create_network, layers

import Distributions, Plots
ENV["GKSwstype"] = "nul"

include("loss_functions.jl")
include("activation_functions.jl")
include("weights_initializers.jl")

struct LayerParameters
    nb_nodes::Int64
    activation::AbstractString
    weights_initializer::Optional{AbstractString}
end
LayerParameters(nb_nodes::Int64, activation::AbstractString) = LayerParameters(nb_nodes, activation, nothing)

mutable struct NetworkLayer
    weights::Matrix{Float64}
    biases::Vector{Float64}
    activation::LayerActivation
end
NetworkLayer(activation::LayerActivation) = NetworkLayer(Matrix(undef, 0, 0), [], activation)

const layers::Vector{NetworkLayer} = []

const cache_layers::Vector{Tuple{Matrix{Float64},Vector{VecOrMat{Float64}}}} = []

function create_network(layers_parameters::AbstractArray{LayerParameters})

    if length(layers_parameters) < 2
        throw(ArgumentError("create_network should take at least 2 layers"))
    end

    empty!(layers)
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

const total_losses::Vector{Float64} = []
const total_valid_losses::Vector{Float64} = []
const total_accuracy::Vector{Float64} = []
const total_valid_accuracy::Vector{Float64} = []

function keep_track_of_loss_and_accuracy(loss, losses, dataset_valid, Y_valid)
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

    Y_preds = predict(convert(Matrix{Float64}, dataset_valid'))
    valid_losses = [loss.fun(Y_preds[:, i], Y_valid'[:, i]) for i = axes(Y_preds, 2)]
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

function predict(A::Matrix{Float64})
    #       r1 r2
    # A = f1
    #     f2
    for layer in layers
        #         wL-1,1 wL-1,2
        # W = wL,1
        #     wL,2
        W = layer.weights


        # B = b1 b2
        B = layer.biases

        #       r1 r2
        # Z = z1
        #     z2
        Z = W * A .+ B
        push!(
            cache_layers,
            (A, [layer.activation.derivative(Zcol) for Zcol in eachcol(Z)])
        )

        A = reduce(hcat, [layer.activation.fun(Zcol) for Zcol in eachcol(Z)])
    end
    return A
end

@views function backpropagation(layer_idx::Int64, ∂L∂𝝈s::Matrix{Float64}, learning_rate::Float64, batch_size::Int64)::Matrix{Float64}
    w_gradient_sum::Matrix{Float64} = zeros(size(layers[layer_idx].weights))
    b_gradient_sum::Vector{Float64} = zeros(size(layers[layer_idx].biases))
    if layer_idx == length(layers)
        Aprevs, d𝝈dZs = pop!(cache_layers)

        ∂L∂Zs::Matrix{Float64} = Matrix(undef, size(∂L∂𝝈s, 1), 0)
        for i = 1:batch_size
            if ndims(d𝝈dZs[i]) == 1
                ∂L∂Zs = hcat(∂L∂Zs, ∂L∂𝝈s[:, i] .* d𝝈dZs[i])
            elseif ndims(d𝝈dZs[i]) == 2
                ∂L∂Zs = hcat(∂L∂Zs, d𝝈dZs[i]' * ∂L∂𝝈s[:, i])
            else
                throw(ErrorException("Error during backpropagation"))
            end
            w_gradient_sum += ∂L∂Zs[:, end] * Aprevs[:, i]'
            b_gradient_sum += ∂L∂Zs[:, end]
        end

        layers[layer_idx].weights -= learning_rate * w_gradient_sum
        layers[layer_idx].biases -= learning_rate * b_gradient_sum

        return ∂L∂Zs
    end

    Wnext::Matrix{Float64} = layers[layer_idx+1].weights
    ∂L∂Zsnext = backpropagation(layer_idx + 1, ∂L∂𝝈s, learning_rate, batch_size)

    Aprevs, d𝝈dZs = pop!(cache_layers)

    ∂L∂Zs = Matrix(undef, size(Wnext, 2), 0)
    for i = 1:batch_size
        if ndims(d𝝈dZs[i]) == 1
            ∂L∂Zs = hcat(∂L∂Zs, Wnext' * ∂L∂Zsnext[:, i] .* d𝝈dZs[i])
        elseif ndims(d𝝈dZs[i]) == 2
            ∂L∂Zs = hcat(∂L∂Zs, d𝝈dZs[i]' * (Wnext' * ∂L∂Zsnext[:, i]))
        else
            throw(ErrorException("Error during backpropagation"))
        end
        w_gradient_sum += ∂L∂Zs[:, end] * Aprevs[:, i]'
        b_gradient_sum += ∂L∂Zs[:, end]
    end

    layers[layer_idx].weights -= learning_rate * w_gradient_sum
    layers[layer_idx].biases -= learning_rate * b_gradient_sum

    return ∂L∂Zs
end

function train(dataset, Y, dataset_valid, Y_valid, loss_fun, learning_rate, batch_size, epochs)::Nothing
    loss = find_loss_function(loss_fun)

    for _ in 1:epochs
        losses = []
        current_idx = 1
        while current_idx < size(Y, 1)
            batch_size = min(batch_size, size(Y, 1) - current_idx)
            X::Matrix{Float64} = dataset[current_idx:current_idx+batch_size-1, :]'
            Y_actual::Matrix{Float64} = Y[current_idx:current_idx+batch_size-1, :]'
            current_idx += batch_size

            Y_pred = predict(X)

            push!(
                losses,
                loss.fun(Y_pred, Y_actual)
            )

            loss_derivatives = loss.derivative(Y_pred, Y_actual)

            backpropagation(1, loss_derivatives, learning_rate, batch_size)
        end
        keep_track_of_loss_and_accuracy(loss, losses, dataset_valid, Y_valid)
    end
    Plots.savefig(Plots.plot([total_losses, total_valid_losses], label=["training loss" "validation loss"]), "loss_plot.png")
    Plots.savefig(Plots.plot([total_accuracy, total_valid_accuracy], label=["training accuracy" "validation accuracy"]), "accuracy_plot.png")
    return nothing
end
end
