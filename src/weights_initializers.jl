
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
