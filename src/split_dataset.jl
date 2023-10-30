import Random

if length(ARGS) != 1
    println("usage: julia --project src/split_dataset data.csv")
    exit(1)
end

function read_dataset_lines(dataset_file)
    dataset_lines = []
    try
        dataset_lines = readlines(dataset_file)
    catch e
        println("bad dataset format: $e")
    end
    return dataset_lines
end

dataset_lines = read_dataset_lines(ARGS[1])
Random.shuffle!(dataset_lines)
training_length = trunc(Int32, length(dataset_lines) * 0.8)

write("dataset_training.csv", join(dataset_lines[1:training_length], "\n"))
write("dataset_test.csv", join(dataset_lines[training_length+1:end], "\n"))
