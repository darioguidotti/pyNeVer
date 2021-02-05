import pynever.networks as networks
import pynever.nodes as nodes
import pynever.utilities as util
import pynever.datasets as dt
import pynever.strategies.training as training
import pynever.strategies.pruning as pruning
import pynever.strategies.conversion as conversion
import copy

# Building of the network of interest
small_net = networks.SequentialNetwork("SmallNetwork")
small_net.add_node(nodes.FullyConnectedNode('Linear_1', 784, 64))
small_net.add_node(nodes.BatchNorm1DNode('BatchNorm_2', 64))
small_net.add_node(nodes.ReLUNode('ReLU_3', 64))
small_net.add_node(nodes.FullyConnectedNode('Linear_4', 64, 32))
small_net.add_node(nodes.BatchNorm1DNode('BatchNorm_5', 32))
small_net.add_node(nodes.ReLUNode('ReLU_6', 32))
small_net.add_node(nodes.FullyConnectedNode('Linear_7', 32, 16))
small_net.add_node(nodes.BatchNorm1DNode('BatchNorm_8', 16))
small_net.add_node(nodes.ReLUNode('ReLU_9', 16))
small_net.add_node(nodes.FullyConnectedNode('Linear_10', 16, 10))

# Loading of the dataset of interest
dataset = dt.MNISTDataset()

# Initialization of the training and pruning parameters
cuda = False  # If possible the experiment should be run with cuda, otherwise it will take quite some time.
epochs = 100
train_batch_size = 128
test_batch_size = 64
learning_rate = 0.1
batch_norm_decay = 0.001
weight_sparsity_rate = 0.7  # Prune 70% of the weights
neuron_sparsity_rate = 0.5  # Prune 50% of the neurons

# Creation of the trainers needed for baseline training and fine tuned pruning.
trainer_wp = training.AdamTraining(epochs, train_batch_size, test_batch_size, learning_rate, l1_decay=0.0001,
                                   cuda=cuda, fine_tuning=True)
trainer_ns = training.AdamTraining(epochs, train_batch_size, test_batch_size, learning_rate, cuda=cuda,
                                   batchnorm_decay=batch_norm_decay, fine_tuning=False)
trainer_baseline = training.AdamTraining(epochs, train_batch_size, test_batch_size, learning_rate, cuda=cuda,
                                         weight_decay=0.0001, fine_tuning=False)

# Training and pruning of the networks of interest
baseline_net = copy.deepcopy(small_net)
baseline_net = trainer_baseline.train(baseline_net, dataset)

sparse_net = copy.deepcopy(small_net)
sparse_net = trainer_ns.train(sparse_net, dataset)
trainer_ns.fine_tuning = True

wp_pruner = pruning.WeightPruning(weight_sparsity_rate, trainer_wp, pre_training=True)
ns_pruner = pruning.NetworkSlimming(neuron_sparsity_rate, trainer_ns, pre_training=False)

wp_pruned_net = copy.deepcopy(small_net)
wp_pruned_net = wp_pruner.prune(wp_pruned_net, dataset)

ns_pruned_net = copy.deepcopy(sparse_net)
ns_pruned_net = ns_pruner.prune(ns_pruned_net, dataset)

baseline_accuracy, baseline_loss = util.testing(conversion.PyTorchConverter().from_neural_network(baseline_net),
                                                dataset, test_batch_size, cuda=cuda)
sparse_accuracy, sparse_loss = util.testing(conversion.PyTorchConverter().from_neural_network(sparse_net),
                                            dataset, test_batch_size, cuda=cuda)
ns_accuracy, ns_loss = util.testing(conversion.PyTorchConverter().from_neural_network(ns_pruned_net),
                                    dataset, test_batch_size, cuda=cuda)
wp_accuracy, wp_loss = util.testing(conversion.PyTorchConverter().from_neural_network(wp_pruned_net),
                                    dataset, test_batch_size, cuda=cuda)


# Batch norm fusion for the networks of interest (needed for verification and abstraction).
com_baseline = util.combine_batchnorm1d_net(baseline_net)
com_sparse_net = util.combine_batchnorm1d_net(sparse_net)
com_wp_pruned_net = util.combine_batchnorm1d_net(wp_pruned_net)
com_ns_pruned_net = util.combine_batchnorm1d_net(ns_pruned_net)

c_baseline_accuracy, c_baseline_loss = util.testing(conversion.PyTorchConverter().from_neural_network(com_baseline),
                                                    dataset, test_batch_size, cuda=cuda)
c_sparse_accuracy, c_sparse_loss = util.testing(conversion.PyTorchConverter().from_neural_network(com_sparse_net),
                                                dataset, test_batch_size, cuda=cuda)
c_ns_accuracy, c_ns_loss = util.testing(conversion.PyTorchConverter().from_neural_network(com_ns_pruned_net),
                                        dataset, test_batch_size, cuda=cuda)
c_wp_accuracy, c_wp_loss = util.testing(conversion.PyTorchConverter().from_neural_network(com_wp_pruned_net),
                                        dataset, test_batch_size, cuda=cuda)

print("ACCURACIES (% of samples correctly classified):\n")
print(f"Baseline: {baseline_accuracy}, Sparse: {sparse_accuracy}, NS: {ns_accuracy}, WP: {wp_accuracy}")
print(f"COMBINED BATCHNORM NETWORKS")
print(f"Baseline: {c_baseline_accuracy}, Sparse: {c_sparse_accuracy}, NS: {c_ns_accuracy}, WP: {c_wp_accuracy}")
