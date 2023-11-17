Optional{T} = Union{Missing,Nothing,T}

function mean_square(Y_preds, Ys)::Float64
    loss = 0
    for i = axes(Y_preds, 2)
        loss += sum((Y_preds[:, i] - Ys[:, i]) .^ 2) / length(Y_preds[:, i])
    end
    return loss / size(Y_preds, 2)
end

function mean_square_derivative(Y_preds, Ys)::Matrix{Float64}

    result = Matrix(undef, size(Y_preds, 1), 0)
    for i = axes(Y_preds, 2)
        result = hcat(result, Y_preds[:, i] - Ys[:, i]) / length(Y_preds[:, i])
    end
    return result
end

function binary_crossentropy(Y_preds, Ys)::Float64
    binary_crossentropy_one_var(y_pred, y) = y == 1 ? log2(y_pred) : log2(1 - y_pred)

    loss = -sum(binary_crossentropy_one_var.(Y_preds, Ys))
    return loss / size(Y_preds, 2)
end

function binary_crossentropy_derivative(Y_preds, Ys)::Matrix{Float64}
    binary_crossentropy_derivative_one_var(y_pred, y) = y == 1 ? -1 / max(y_pred, 1e-300) : 1 / 1 - min(y_pred, 1 - 1e-300)

    result = Matrix(undef, size(Y_preds, 1), 0)
    @views for i = axes(Y_preds, 2)
        result = hcat(result, binary_crossentropy_derivative_one_var.(Y_preds[:, i], Ys[:, i]))
    end
    return result
end

function crossentropy(Y_preds, Ys)::Float64
    loss = 0
    for i = axes(Y_preds, 2)
        loss += -sum(Ys[:, i] .* log2.(Y_preds[:, i]))
    end
    return loss / size(Y_preds, 2)
end

function crossentropy_derivative(Y_preds, Ys)::Matrix{Float64}
    result = Matrix(undef, size(Y_preds, 1), 0)
    for i = axes(Y_preds, 2)
        -Ys[:, i] ./ Y_preds[:, i]
    end
    return result
end

function find_loss_function(str)::Loss
    losses = Dict(
        [
        "mean_square" => Loss(mean_square, mean_square_derivative),
        "binary_crossentropy" => Loss(binary_crossentropy, binary_crossentropy_derivative, 1.0 * length(layers[end].biases)),
        "crossentropy" => Loss(crossentropy, crossentropy_derivative, 1.0),
    ]
    )
    return losses[str]
end

struct Loss
    fun::Union{typeof(mean_square),typeof(binary_crossentropy),typeof(crossentropy)}
    derivative::Union{typeof(mean_square_derivative),typeof(binary_crossentropy_derivative),typeof(crossentropy_derivative)}
    threshold::Optional{Float64}
    Loss(fun, derivative, threshold=nothing) = new(fun, derivative, threshold)
end
