import pynever.networks as networks
import pynever.nodes as nodes
import numpy as np
import pynever.strategies.training as train
import pynever.strategies.verification as ver
import pynever.strategies.conversion as conv
import pynever.datasets as dt
import os
import time
import onnx

n_epochs = 50
train_batch_size = 64
test_batch_size = 32
learning_rate = 0.001

input_size = 8
output_size = 6
dataset = dt.DynamicsJamesPos("data/James/james_pos_train.txt", "data/James/james_pos_test.txt")

parameter_sets = [[[5, 0.25, 0], [5, 0.5, 0], [5, 1.0, 0], [5, 0.25, 1], [5, 0.5, 1], [5, 1.0, 1],
                   [5, 0.25, 2], [5, 0.5, 2], [5, 1.0, 2]],
                  [[10, 0.25, 0], [10, 0.5, 0], [10, 1.0, 0]],
                  [[15, 0.25, 0], [15, 0.5, 0], [15, 1.0, 0]],
                  [[20, 0.25, 0], [20, 0.5, 0]]]


for p_set_hn in parameter_sets:

    num_hidden_neurons = p_set_hn[0][0]
    net_id = f"James_Net_HN={num_hidden_neurons}"
    if os.path.exists("james_net/" + net_id + ".onnx"):
        net = conv.ONNXConverter().to_neural_network(conv.ONNXNetwork(net_id,
                                                                      onnx.load("james_net/" + net_id + ".onnx")))
    else:
        net = networks.SequentialNetwork("James_Net")
        net.add_node(nodes.FullyConnectedNode("FullyConnected1", input_size, num_hidden_neurons))
        net.add_node(nodes.SigmoidNode("Sigmoid1", num_hidden_neurons))
        net.add_node(nodes.FullyConnectedNode("FullyConnected2", num_hidden_neurons, output_size))

        training_strategy = train.AdamTrainingRegression(n_epochs, train_batch_size, test_batch_size, learning_rate)

        net = training_strategy.train(net, dataset)
        onnx.save(conv.ONNXConverter().from_neural_network(net).onnx_network, "james_net/" + net_id + ".onnx")

    out_pred_mat = []
    out_pred_bias = []
    for i in range(output_size):
        temp_mat = np.zeros((1, output_size))
        temp_bias = np.zeros((1, 1))
        temp_bias[0, 0] = -1
        temp_mat[0, i] = 1

        out_pred_mat.append(temp_mat)
        out_pred_bias.append(temp_bias)

        temp_mat = np.zeros((1, output_size))
        temp_bias = np.zeros((1, 1))
        temp_bias[0, 0] = -1
        temp_mat[0, i] = -1

        out_pred_mat.append(temp_mat)
        out_pred_bias.append(temp_bias)

    for p_set in p_set_hn:

        bound = p_set[1]
        in_pred_mat = []
        in_pred_bias = []
        for i in range(input_size):
            temp_lb = np.zeros(input_size)
            temp_lb[i] = -1
            temp_ub = np.zeros(input_size)
            temp_ub[i] = 1
            in_pred_mat.append(temp_lb)
            in_pred_mat.append(temp_ub)
            in_pred_bias.append([bound])
            in_pred_bias.append([bound])

        in_pred_mat = np.array(in_pred_mat)
        in_pred_bias = np.array(in_pred_bias)

        prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)

        master_log_file = "logs/JamesExperimentLog.txt"

        neuron_relevance = False
        ref_percentage = 0
        thresholding = False
        iqr_mult = 0
        ref_level = p_set[2]

        log_filepath = f"logs/James_NET=H{num_hidden_neurons}_BOUND={bound}_P=Global_REF={ref_level}.txt"

        print(f"James_NET=H{num_hidden_neurons}_BOUND={bound}_P=Global_REF={ref_level}")

        verifier = ver.NeverVerification(log_filepath, neuron_relevance, thresholding, iqr_mult, ref_percentage,
                                         ref_level)

        time_start = time.perf_counter()
        safe = verifier.verify(net, prop)
        time_end = time.perf_counter()
        print(f"SAFETY: {safe}")

        with open(master_log_file, "a") as master_log:
            master_log.write(
                f"James,Global,{ref_level},{num_hidden_neurons},{bound},{safe},{time_end - time_start}\n")

