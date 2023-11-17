softmax(Z)::Vector{Float64} = begin
    s = sum(2 .^ Z)
    return (2 .^ Z) ./ s
end

softmax_derivative(Z)::Matrix{Float64} = begin
    sZ = softmax(Z)
    jacobian = sZ * -sZ'
    for i in eachindex(sZ)
        jacobian[i, i] = sZ[i] * (1 - sZ[i])
    end
    return jacobian
end

sigmoid(Z)::Vector{Float64} = 1 ./ (1 .+ exp2.(-Z))
sigmoid_derivative(Z)::Vector{Float64} = sigmoid(Z) .* (1 .- sigmoid(Z))

relu(Z)::Vector{Float64} = max.(0, Z)
relu_derivative(Z)::Vector{Float64} = [z > 0 ? 1 : 0 for z in Z]

leaky_relu(Z)::Vector{Float64} = [z > 0 ? z : -0.01z for z in Z]
leaky_relu_derivative(Z)::Vector{Float64} = [z > 0 ? 1 : -0.01 for z in Z]

tanh(Z)::Vector{Float64} = (exp2.(Z) - exp2.(-Z)) ./ (exp2.(Z) + exp2.(-Z))
tanh_derivative(Z)::Vector{Float64} = 1 .- tanh.(Z) .^ 2

struct LayerActivation
    fun::Union{typeof(softmax),typeof(sigmoid),typeof(relu),typeof(leaky_relu),typeof(tanh)}
    derivative::Union{typeof(softmax_derivative),typeof(sigmoid_derivative),typeof(relu_derivative),typeof(leaky_relu_derivative),typeof(tanh_derivative)}
end

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
