using DataFrames
using CSV
using Random
using StatsBase
using MLJ

mutable struct Node
    ids # index of data at this node
    entropy
    split_attribute
    children
    split_value
    label
    depth
end

mutable struct Tree
    root
    size_of_train
    data
    attributes # contains all attributes of training data
    target # contains all labels in the training set
    min_gain
    max_depth
    min_samples_split # minimum number of data samples in a node
end

function initialize_node(ids=[], entropy=0, children=[], depth=0)
    """Initialize a node."""
    node = Node(ids, entropy, nothing, children, nothing, nothing, depth)
    return node
end

function initialize_tree(max_depth=10, min_samples_split=2, min_gain=1e-4)
    """Initialize a tree."""
    tree = Tree(nothing, 0, nothing, nothing, nothing, min_gain, max_depth, min_samples_split)
    return tree
end

function set_attribute(node::Node, split_attribute, split_value)
    """Set the split attribute and split value to a node."""
    node.split_attribute = split_attribute
    node.split_value = split_value
end

function set_label(node::Node, target)
    """Set the label to a node by majority voting."""
    data = target[node.ids]
    mode_value = mode(data)
    node.label = mode_value
end

function entropy(ids, target)
    """Calculate the entropy of a given set of target values based on the provided indices (ids array)."""
    if length(ids) == 0
        return 0
    else
        data = target[ids]
        freq_dict = countmap(data)
        freq_array = collect(values(freq_dict))
        if length(freq_array) == 1
            return 0
        end
        return entropy(freq_array)
    end
end

function entropy(freq::Array)
    """Calculate the entropy based on the given frequency array."""
    freq_0 = freq[freq .!= 0]
    prob_0 = freq_0 / sum(freq_0)
    return -sum(prob_0 .* log.(prob_0))
end

function split_by_threshold(threshold, ids, data)
    """Split a set of indices based on a given threshold and corresponding data values."""
    ids_of_less = []
    ids_of_greater = []
    for i in ids
        if isless(data[i], threshold)
            push!(ids_of_less, i)
        else
            push!(ids_of_greater, i)
        end
    end
    return ids_of_less, ids_of_greater
end

function split(node::Node, tree::Tree)
    """Split a decision tree node based on the best attribute and value, maximizing information gain."""
    ids = node.ids
    best_gain = 0.0
    best_HxS = Inf
    best_splits = []
    best_attribute = nothing
    split_value = 0
    data = tree.data
    attributes = tree.attributes
    target = tree.target
    for (i, attribute) in enumerate(attributes)
        values = unique(data[ids, i]) # get unique values
        if length(values) == 1
            continue
        end

        for value in values
            splits = []
            # split the data into 2 parts based on the threshold = `value`
            ids_of_less, ids_of_greater = split_by_threshold(value, ids, data[:, attribute])
            push!(splits, ids_of_less)
            push!(splits, ids_of_greater)
            # do not split if this node has a small number of data samples
            if minimum(map(length, splits)) < tree.min_samples_split
                continue
            end
            # calculate the information gain
            HxS = 0.0
            for split in splits
                HxS += length(split)*entropy(split, target)/length(ids)
            end
            gain = node.entropy - HxS
            # do not split if the gain is too small
            if gain < tree.min_gain
                continue
            end
            # if this information gain is the biggest (smallest entropy), update some important variables
            if gain > best_gain
                best_gain = gain
                best_splits = splits
                best_attribute = attribute
                split_value = value
            end
        end
    end
    set_attribute(node, best_attribute, split_value) # set the split attribute and split value to this node
    # create child nodes of this node and return them
    child_nodes = []
    for split in best_splits
        push!(child_nodes, initialize_node(split, entropy(split, target), [], node.depth + 1))
    end
    return child_nodes
end

function calculate_accuracy(y_true, y_pred)
    """Calculate the accuracy of predicted values compared to true values."""
    correct_predictions = sum(y_true .== y_pred)
    total_samples = length(y_true)
    accuracy_value = correct_predictions / total_samples
    return accuracy_value
end

function fit(tree::Tree, data, attributes, target)
    """Fit a decision tree to the provided training data."""
    tree.size_of_train = size(data, 1)
    tree.data = data
    tree.attributes = attributes
    tree.target = target
    ids = range(1, tree.size_of_train)
    tree.root = initialize_node(ids, entropy(ids, tree.target), [], 0)
    queue = [tree.root]
    while length(queue) != 0
        node = popfirst!(queue)
        # iterate until the tree has too much depth
        # or the split does not reduce the entropy too much
        # (information gain is less than the minimum threshold)
        if node.depth < tree.max_depth || node.entropy >= tree.min_gain
            node.children = split(node, tree)
            if length(node.children) == 0 # if this is a leaf node, set the label
                set_label(node, tree.target)
            end
            append!(queue, node.children)
        else
            set_label(node, tree.target)
        end
    end
end

function predict(tree::Tree, new_data)
    """Predict the target labels for new data using a fitted decision tree."""
    size_of_data = size(new_data, 1)
    labels = []
    for i = 1:size_of_data
        x = new_data[i, :]
        node = tree.root # traverse from the root to the leaf node
        while length(node.children) != 0
            if x[node.split_attribute] < node.split_value || length(node.children) == 1
                node = node.children[1]
            else
                node = node.children[2]
            end
        end
        push!(labels, node.label)
    end
    return labels
end

function print_tree(tree::Tree)
    queue = [tree.root]
    while length(queue) != 0
        node = popfirst!(queue)
        println("\nNode: ", node)
        append!(queue, node.children)
    end
end

function main()
    df = DataFrame(CSV.File("iris.csv")) # read data from a CSV file and convert it to a DataFrame
    attributes = names(df)[1:end-1] # get the attribute array
    train_df, test_df = MLJ.partition(df, 2/3, shuffle=true) # randomly split data into training and testing sets
    target = train_df[!, end] # get the training labels
    test_labels = test_df[!, end] # get the testing labels
    train_df = train_df[!, 1:size(df, 2)-1] # get the training data
    test_df = test_df[!, 1:size(df, 2)-1] # get the testing data
    tree = initialize_tree(3) # initialize a decision tree with the specific depth
    fit(tree, train_df, attributes, target) # fit the training data to the decision tree
    predicted_labels = predict(tree, test_df) # use the fitted tree to predict the testing data
    accuracy = calculate_accuracy(test_labels, predicted_labels) # evaluate the fitted tree with the accuracy metric
    println("The accuracy of the fitted decision tree on the testing set is ", accuracy) # print the result
    # print_tree(tree)
end

main()