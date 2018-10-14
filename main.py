import pickle
import numpy as np
import pandas as pd

# Observations:
#   Too high nodes, higher error,
#   Lower error margin, is better but longer training
# 'layers': [9, 11, 7]
architecture = {
    'inputs': 354,
    'layers': [16, 7],
    'outputs': 8,
    'eta': 0.1,
    'error_margin': 0.00002
}


def dump_pickle(path):
    with open(path, 'rb') as handle:
        print(pickle.load(handle)['architecture'])


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_prime(y: np.ndarray) -> np.ndarray:
    return np.multiply(y, (1-y))


def generate_network(architecture):
    nn_inputs = np.zeros((architecture['inputs'], 1))

    nn_weights = []
    # Append inputs in the end so that we get that on -1 index
    layers = architecture['layers'] + [architecture['inputs']]
    for index, layer_node_count in enumerate(layers[:-1]):
        prev_input = layers[index-1]
        nn_weights.append(
            np.random.normal(
                loc=0,
                scale=np.sqrt(layer_node_count),
                size=(layer_node_count, prev_input)
            )
        )

    nn_biases = []
    for layer_node_count in architecture['layers']:
        nn_biases.append(
            np.random.normal(
                loc=0,
                scale=np.sqrt(layer_node_count),
                size=(layer_node_count, 1)
            )
        )

    nn_output_weights = np.random.normal(
        loc=0,
        scale=np.sqrt(architecture['outputs']),
        size=(architecture['outputs'], architecture['layers'][-1])
    )
    nn_output_biases = np.random.normal(
        loc=0,
        scale=np.sqrt(architecture['outputs']),
        size=(architecture['outputs'], 1)
    )
    nn_outputs = np.zeros((architecture['outputs'], 1))

    return {
        'inputs': nn_inputs,
        'layer_weights': nn_weights,
        'layer_biases': nn_biases,
        'output_weights': nn_output_weights,
        'output_biases': nn_output_biases,
        'outputs': nn_outputs,
    }

training_set_count = 8
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
Y = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
])


# Returns trained weights and biases
def train(
    eta,
    network,
    training_set_count,
    inputs,
    labels,
    epoch=100000,
    error_margin=0.00002
):
    total_error = np.zeros((epoch, 1))
    for iteration in range(epoch):
        training_set_order = np.random.permutation(training_set_count)
        for _, training_input_index in enumerate(training_set_order):
            training_input = X[training_input_index][None].T
            expected_output = Y[training_input_index][None].T

            # Forward Pass
            # -1 in ys is the output
            # input, layer_1, layer_2, output
            ys = [training_input]
            weights = network['layer_weights'] + [network['output_weights']]
            biases = network['layer_biases'] + [network['output_biases']]
            for layer, layer_weight in enumerate(weights):
                v = (
                    np.dot(layer_weight, ys[layer]) +
                    biases[layer]
                )
                y = logistic(v)
                ys.append(y)

            # Back Propagation
            error = expected_output - ys[-1]
            delta_out = np.multiply(error, logistic_prime(ys[-1]))

            # out, layer_2, layer_1
            deltas = [delta_out]
            reversed_weights = (
                [network['output_weights']] +
                list(reversed(network['layer_weights']))
            )

            # We traverse the weights from output back to first hidden layer
            # But only consider weights of output to second to the last layer
            for rev_layer, layer_weight in enumerate(reversed_weights[:-1]):
                delta = np.multiply(
                    np.dot(layer_weight.T, deltas[rev_layer]),
                    logistic_prime(ys[(-2) + (-rev_layer)])
                )
                deltas.append(delta)

            # Update weights using deltas computed
            network['output_weights'] += np.multiply(
                eta,
                np.dot(deltas[0], ys[-2].T)
            )

            # Start from layer 1 to layer n
            for layer, _ in enumerate(network['layer_weights']):
                network['layer_weights'][layer] += np.multiply(
                    eta,
                    np.dot(deltas[(-1) + (-layer)], ys[layer].T)
                )

            # Update biases using deltas computed
            network['output_biases'] += np.multiply(
                eta,
                deltas[0]
            )

            # Start from layer 1 to layer n
            for layer, _ in enumerate(network['layer_biases']):
                network['layer_biases'][layer] += np.multiply(
                    eta,
                    deltas[(-1) + (-layer)]
                )
        total_error[iteration] += np.sum(np.multiply(error, error))
        if iteration % 500 == 0:
            print('Iteration: {} Error: {}'.format(
                iteration,
                total_error[iteration])
            )

        if total_error[iteration] < error_margin:
            print('Final Error: {}'.format(total_error[iteration]))
            break


def test(network, inputs, labels):
    total_run = 0
    correct_prediction = 0
    for index, entry in enumerate(inputs):
        network_input = entry[None].T
        expected_output = labels[index].T

        # Forward Pass
        # -1 in ys is the output
        # input, layer_1, layer_2, output
        ys = [network_input]
        weights = network['layer_weights'] + [network['output_weights']]
        biases = network['layer_biases'] + [network['output_biases']]
        for layer, layer_weight in enumerate(weights):
            v = (
                np.dot(layer_weight, ys[layer]) +
                biases[layer]
            )
            y = logistic(v)
            ys.append(y)

        total_run += 1
        prediction = np.array([int(output >= 0.5) for output in ys[-1]])
        # print('Prediction: ', prediction)
        # print('Expected: ', expected_output)
        # print('*'*50)
        if np.array_equal(prediction, expected_output):
            correct_prediction += 1
    accuracy = correct_prediction/(total_run*1.0) * 100
    print('Accuracy {}'.format(accuracy))
    return accuracy


# Label Frequency Count
# 1    1625
# 4     483
# 8     466
# 6     310
# 5     287
# 2     233
# 7      52
# 3      30

# Attempt 1:
# We'll have 20 * 8 training data = 160
# Validation set would be remaining set
def partition_dataset_attempt_1(n=20):
    '''
        Reads in data and data_labels and produces the ff:
        training values and labels
        validation values and labels
    '''
    # Load data set
    dataset = pd.read_csv(
        './CS280_PA2_2018/data.csv',
        names=['input_{}'.format(index) for index in range(354)]
    )

    # Load data labels
    labels = pd.read_csv(
        './CS280_PA2_2018/data_labels.csv',
        names=['label']
    )

    # Concate the two data set
    data_frame = pd.concat([dataset, labels], axis=1, ignore_index=True)

    # Sample each category
    training_set = pd.concat(
        [data.sample(n=n) for _, data in data_frame.groupby(354)]
    )

    # Retrieve the rest and generate validation csv files
    validation_set = data_frame.merge(
        training_set,
        on=[index for index in range(355)],
        how='left',
        indicator=True
    )
    validation_set = validation_set[
        validation_set['_merge'] == 'left_only'
    ]
    validation_set.pop('_merge')

    # Generate validation csv files
    validation_labels = validation_set.pop(354)
    validation_labels.to_csv(
        './processed_data/validation_labels.csv',
        index=False,
        header=False
    )
    validation_set.to_csv(
        './processed_data/validation_set.csv',
        index=False,
        header=False
    )

    # Generate training csv files
    training_labels = training_set.pop(354)
    training_labels.to_csv(
        './processed_data/training_labels.csv',
        index=False,
        header=False
    )
    training_set.to_csv(
        './processed_data/training_set.csv',
        index=False,
        header=False
    )


def get_training_set():
    dataset = pd.read_csv(
        './processed_data/training_set.csv',
        names=['input_{}'.format(index) for index in range(354)]
    )
    labels = pd.read_csv(
        './processed_data/training_labels.csv',
        names=['labels']
    )

    true_label = []
    for _, label in labels.iterrows():
        label_arr = [0] * 8
        label_arr[label[0] - 1] = 1
        true_label.append(label_arr)

    return (
        dataset.shape[0],
        dataset.values,
        np.array(true_label)
    )


def get_validation_set():
    dataset = pd.read_csv(
        './processed_data/validation_set.csv',
        names=['input_{}'.format(index) for index in range(354)]
    )
    labels = pd.read_csv(
        './processed_data/validation_labels.csv',
        names=['labels']
    )

    true_label = []
    for _, label in labels.iterrows():
        label_arr = [0] * 8
        label_arr[label[0] - 1] = 1
        true_label.append(label_arr)

    return (
        dataset.values,
        np.array(true_label)
    )

if __name__ == '__main__':
    # partition_dataset_attempt_1(n=25)

    (N, X, Y) = get_training_set()
    (X_val, Y_val) = get_validation_set()
    for i in range(10):
        print('Epoch: {}'.format(i))
        network = generate_network(architecture)

        train(
            eta=architecture['eta'],
            network=network,
            training_set_count=N,
            inputs=X,
            labels=Y,
            error_margin=architecture['error_margin']
        )

        accuracy = test(
            network=network,
            inputs=X_val,
            labels=Y_val
        )

        if accuracy >= 55:
            # Save network just in case
            with open(
                './processed_data/network_{}.pickle'.format(round(accuracy, 2)),
                'wb'
            ) as handle:
                pickle.dump(
                    {
                        'architecture': architecture,
                        'network': network
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
