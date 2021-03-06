import pynever.utilities as utilities
import numpy as np
import pynever.strategies.verification as ver
import pynever.nodes as nodes
import pynever.networks as networks
import time

property_ids = ["P3", "P4"]

unsafe_mats = [[[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]],
               [[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]]]
unsafe_vecs = [[[0], [0], [0], [0]], [[0], [0], [0], [0]]]

input_lb = [[1500, -0.06, 3.1, 980, 960], [1500, -0.06, 3.1, 1000, 700]]
input_ub = [[1800, 0.06, 3.14, 1200, 1000], [1800, 0.06, 3.14, 1200, 800]]

networks_ids = [["1_1", "1_3", "2_3", "4_3", "5_1"], ["1_1", "1_3", "3_2", "4_2"]]

master_log_file = "logs/ACASXUExperimentLog.txt"

verification_parameters = [[False, 0, False, 0], [True, 1, False, 0], [False, 0.1, False, 0]]
param_set_id = ["Over-Approx", "Complete", "Mixed"]

with open(master_log_file, "a") as master_log:
    master_log.write(
        f"Dataset,NetworkID,PropertyID,Methodology,Safety,Time\n")

for i in range(0, len(property_ids)):

    for j in range(len(networks_ids[i])):

        # Loading of the values of interest of the corresponding ACAS XU network.
        print(f"Verifying {property_ids[i]} on Network {networks_ids[i][j]}.")
        weights, biases, inputMeans, inputRanges, outputMean, outputRange = \
            utilities.parse_nnet(f"nnet/ACASXU_experimental_v2a_{networks_ids[i][j]}.nnet")

        # Creation of the matrixes defining the input set (i.e., in_pred_mat * x <= in_pred_bias).

        # Normalization of the lb and ub.
        norm_input_lb = []
        norm_input_ub = []
        for k in range(len(input_lb[i])):
            norm_input_lb.append((input_lb[i][k] - inputMeans[k]) / inputRanges[k])
            norm_input_ub.append((input_ub[i][k] - inputMeans[k]) / inputRanges[k])

        # Matrixes Creation.
        in_pred_mat = []
        in_pred_bias = []
        for k in range(len(norm_input_lb)):
            lb_constraint = np.zeros(len(norm_input_lb))
            ub_constraint = np.zeros(len(norm_input_ub))
            lb_constraint[k] = -1
            ub_constraint[k] = 1
            in_pred_mat.append(lb_constraint)
            in_pred_mat.append(ub_constraint)
            in_pred_bias.append([-norm_input_lb[k]])
            in_pred_bias.append([norm_input_ub[k]])

        in_pred_bias = np.array(in_pred_bias)
        in_pred_mat = np.array(in_pred_mat)

        # Creation of the matrixes defining the negation of the wanted property (i.e., unsafe region)
        # (i.e., out_pred_mat * y <= out_pred_bias).
        out_pred_mat = np.array(unsafe_mats[i])
        if property_ids[i] == "Property 1":
            out_pred_bias = (np.array(unsafe_vecs[i]) - outputMean) / outputRange
        else:
            out_pred_bias = np.array(unsafe_vecs[i])

        # Construction of our internal representation for the ACAS network.

        network = networks.SequentialNetwork(f"ACAS_XU_{networks_ids[i][j]}")
        for k in range(len(weights)):

            new_fc_node = nodes.FullyConnectedNode(f"FC_{k}", weights[k].shape[1], weights[k].shape[0], weights[k],
                                                   biases[k])
            network.add_node(new_fc_node)

            if k < len(weights) - 1:
                new_relu_node = nodes.ReLUNode(f"ReLU_{k}", weights[k].shape[0])
                network.add_node(new_relu_node)

        # Verification of the network of interest for the property of interest
        prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [out_pred_mat], [out_pred_bias])

        for k in range(len(verification_parameters)):

            p_set = verification_parameters[k]
            neuron_relevance = p_set[0]
            ref_percentage = p_set[1]
            thresholding = p_set[2]
            iqr_mult = p_set[3]
            net_id = networks_ids[i][j]
            p_id = property_ids[i]

            log_filepath = f"logs/ACASXU_NET={net_id}_PROPERTY={property_ids[i]}_{param_set_id[k]}.txt"

            print(f"ACASXU_{net_id}_P={property_ids[i]}_{param_set_id[k]}")

            verifier = ver.NeverVerification(log_filepath, neuron_relevance, thresholding, iqr_mult, ref_percentage, 0)

            time_start = time.perf_counter()
            safe = verifier.verify(network, prop)
            time_end = time.perf_counter()
            print(f"SAFETY: {safe}")

            with open(master_log_file, "a") as master_log:
                master_log.write(
                    f"ACASXU,{net_id},{property_ids[i]},{param_set_id[k]},{safe},{time_end - time_start}\n")
