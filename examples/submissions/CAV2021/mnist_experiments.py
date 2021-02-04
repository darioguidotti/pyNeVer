import onnx
import numpy as np
import time
import pynever.strategies.conversion as conv
import pynever.datasets as dt
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.nodes as nodes

mnist = dt.MNISTDataset()
datas, targets = mnist.get_test_set()

net_ids = ["A_NS_SET1.onnx", "A_NS_SET2.onnx", "A_NS_SET3.onnx",
           "B_NS_SET1.onnx", "B_NS_SET2.onnx", "B_NS_SET3.onnx",
           "A_Baseline_SET1.onnx", "A_Baseline_SET2.onnx", "A_Baseline_SET3.onnx",
           "B_Baseline_SET1.onnx", "B_Baseline_SET2.onnx", "B_Baseline_SET3.onnx"]

"""net_ids = ["Small_NS_SET1.onnx", "Small_NS_SET2.onnx", "Small_NS_SET3.onnx",
           "Small_WP_SET1.onnx", "Small_WP_SET2.onnx", "Small_WP_SET3.onnx",
           "Small_Baseline_SET1.onnx", "Small_Baseline_SET2.onnx", "Small_Baseline_SET3.onnx"]"""

data_index = 10
data = datas[data_index]
target = targets[data_index]
adversarial_target = 0
adversarial_magnitude = 0.1

data_size = 784
in_pred_mat = []
in_pred_bias = []
for i in range(data_size):

    lb_constraint = np.zeros(data_size)
    ub_constraint = np.zeros(data_size)
    lb_constraint[i] = -1
    ub_constraint[i] = 1
    in_pred_mat.append(lb_constraint)
    in_pred_mat.append(ub_constraint)
    if data[i] - adversarial_magnitude < -1:
        in_pred_bias.append([1])
    else:
        in_pred_bias.append([-(data[i] - adversarial_magnitude)])

    if data[i] + adversarial_magnitude > 1:
        in_pred_bias.append([1])
    else:
        in_pred_bias.append([data[i] + adversarial_magnitude])

in_pred_bias = np.array(in_pred_bias)
in_pred_mat = np.array(in_pred_mat)

num_output = 10
out_pred_mat = []
for i in range(num_output):
    if i != adversarial_target:
        temp = np.zeros(num_output)
        temp[adversarial_target] = 1
        temp[i] = -1
        out_pred_mat.append(temp)

out_pred_mat = [np.array(out_pred_mat)]
out_pred_bias = [np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0]])]

lr_property = ver.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)

master_log_file = "logs/MNISTExperimentLog.txt"

ver_param_sets = [[False, 0, False, 0]]
param_set_id = ["Over-Approx"]

with open(master_log_file, "a") as master_log:
    master_log.write(f"Dataset,NetworkID,Methodology,Safety,Time\n")

for net_id in net_ids:

    net_path = "mnist_nets/" + net_id
    net = conv.ONNXConverter().to_neural_network(conv.ONNXNetwork(net_id, onnx.load(net_path)))
    assert isinstance(net, networks.SequentialNetwork)
    current_node = net.get_first_node()
    while current_node is not None:
        if isinstance(current_node, nodes.FullyConnectedNode):
            current_node.weight = np.where(np.abs(current_node.weight) < 0.000001, 0, current_node.weight)
        current_node = net.get_next_node(current_node)

    for p_set in ver_param_sets:

        log_filepath = f"logs/MNIST_NET={net_id}_{param_set_id[0]}.txt"

        print(f"MNIST_NET={net_id}_{param_set_id[0]}")
        verifier = ver.NeverVerification(log_filepath, neuron_relevance=p_set[0], iqr_thresholding=p_set[2],
                                         iqr_mult=p_set[3], refinement_percentage=p_set[1], refinement_level=0)

        time_start = time.perf_counter()
        safe = verifier.verify(net, lr_property)
        time_end = time.perf_counter()
        print(f"SAFETY: {safe}")

        with open(master_log_file, "a") as master_log:
            master_log.write(f"MNIST,{net_id},{param_set_id[0]},{safe},{time_end-time_start}\n")

