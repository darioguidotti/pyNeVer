import copy
import abc
import pynever.networks as networks
import pynever.nodes as nodes
import torch
import onnx
import onnx.numpy_helper


class AlternativeRepresentation(abc.ABC):
    """
    An abstract class used to represent an alternative representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    up_to_date : bool, optional
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """
    def __init__(self, identifier: str, up_to_date: bool = True):
        self.identifier = identifier
        self.up_to_date = up_to_date


class ONNXNetwork(AlternativeRepresentation):
    """
    An class used to represent a ONNX representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    onnx_network : onnx.ModelProto
        Real ONNX network.
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """

    def __init__(self, identifier: str, onnx_network: onnx.ModelProto, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)
        self.onnx_network = copy.deepcopy(onnx_network)


class PyTorchNetwork(AlternativeRepresentation):
    """
    An class used to represent a PyTorch representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    pytorch_network : torch.nn.Module
        Real PyTorch network.
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """
    def __init__(self, identifier: str, pytorch_network: torch.nn.Module, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)
        self.pytorch_network = copy.deepcopy(pytorch_network)


class TensorflowNetwork(AlternativeRepresentation):
    """
    An class used to represent a Tensorflow representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """
    def __init__(self, identifier: str, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)


class ConversionStrategy(abc.ABC):
    """
    An abstract class used to represent a Conversion Strategy.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to an alternative representation determined in the concrete children.
    to_neural_network(AlternativeRepresentation)
        Convert the alternative representation of interest to our internal representation of a Neural Network.

    """

    @abc.abstractmethod
    def from_neural_network(self, network: networks.NeuralNetwork) -> AlternativeRepresentation:
        """
        Convert the neural network of interest to an alternative representation determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        AlternativeRepresentation
            The alternative representation resulting from the conversion of the original network.
        """
        pass

    @abc.abstractmethod
    def to_neural_network(self, alt_rep: AlternativeRepresentation) -> networks.NeuralNetwork:
        """
        Convert the alternative representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : AlternativeRepresentation
            The Alternative Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of Alternative Representation.
        """
        pass


class ONNXConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for ONNX models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a ONNXNetwork model.
    to_neural_network(ONNXNetwork)
        Convert the ONNXNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> ONNXNetwork:
        """
        Convert the neural network of interest to a ONNX representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        ONNXNetwork
            The ONNX representation resulting from the conversion of the original network.

        """

        alt_net = None
        for alt_rep in network.alt_rep_cache:
            if isinstance(alt_rep, ONNXNetwork) and alt_rep.up_to_date:
                alt_net = alt_rep

        if alt_net is None:

            if not network.up_to_date:

                for alt_rep in network.alt_rep_cache:

                    if alt_rep.up_to_date:

                        if isinstance(alt_rep, PyTorchNetwork):
                            pytorch_cv = PyTorchConverter()
                            network = pytorch_cv.to_neural_network(alt_rep)

                        else:
                            raise NotImplementedError
                        break

            if isinstance(network, networks.SequentialNetwork):

                current_node = None
                previous_output = 'X'
                input_info = []
                output_info = []
                initializers = []
                onnx_nodes = []

                while network.get_next_node(current_node) is not None:

                    current_node = network.get_next_node(current_node)
                    current_input = previous_output

                    if network.get_next_node(current_node) is None:
                        current_output = 'Y'
                    else:
                        current_output = current_node.identifier + '_output'

                    if isinstance(current_node, nodes.ReLUNode):

                        input_size = output_size = current_node.num_features
                        input_value_info = onnx.helper.make_tensor_value_info(current_input, onnx.TensorProto.FLOAT,
                                                                              [1, input_size])
                        output_value_info = onnx.helper.make_tensor_value_info(current_output, onnx.TensorProto.FLOAT,
                                                                               [1, output_size])

                        onnx_node = onnx.helper.make_node(
                            'Relu',
                            inputs=[current_input],
                            outputs=[current_output],
                        )

                        input_info.append(input_value_info)
                        output_info.append(output_value_info)
                        onnx_nodes.append(onnx_node)

                    elif isinstance(current_node, nodes.SigmoidNode):

                        input_size = output_size = current_node.num_features
                        input_value_info = onnx.helper.make_tensor_value_info(current_input, onnx.TensorProto.FLOAT,
                                                                              [1, input_size])
                        output_value_info = onnx.helper.make_tensor_value_info(current_output, onnx.TensorProto.FLOAT,
                                                                               [1, output_size])

                        onnx_node = onnx.helper.make_node(
                            'Sigmoid',
                            inputs=[current_input],
                            outputs=[current_output],
                        )

                        input_info.append(input_value_info)
                        output_info.append(output_value_info)
                        onnx_nodes.append(onnx_node)

                    elif isinstance(current_node, nodes.FullyConnectedNode):

                        input_size = current_node.in_features
                        output_size = current_node.out_features

                        input_weight = current_node.identifier + "_weight"
                        input_bias = current_node.identifier + "_bias"

                        input_value_info = onnx.helper.make_tensor_value_info(current_input, onnx.TensorProto.FLOAT,
                                                                              [1, input_size])
                        output_value_info = onnx.helper.make_tensor_value_info(current_output, onnx.TensorProto.FLOAT,
                                                                               [1, output_size])
                        weight_value_info = onnx.helper.make_tensor_value_info(input_weight, onnx.TensorProto.FLOAT,
                                                                               [output_size, input_size])
                        bias_value_info = onnx.helper.make_tensor_value_info(input_bias, onnx.TensorProto.FLOAT,
                                                                             [output_size])

                        # N.B: The Marabou procedure for reading ONNX models do not consider the attributes
                        # transA and transB, therefore we need to transpose the weight vector.
                        weight_tensor = onnx.numpy_helper.from_array(current_node.weight.T, input_weight)
                        bias_tensor = onnx.numpy_helper.from_array(current_node.bias, input_bias)

                        onnx_node = onnx.helper.make_node(
                            'Gemm',
                            inputs=[current_input, input_weight, input_bias],
                            outputs=[current_output],
                            alpha=1.0,
                            beta=1.0,
                            transA=0,
                            transB=0
                        )

                        input_info.append(input_value_info)
                        input_info.append(weight_value_info)
                        input_info.append(bias_value_info)

                        output_info.append(output_value_info)

                        initializers.append(weight_tensor)
                        initializers.append(bias_tensor)

                        onnx_nodes.append(onnx_node)

                    elif isinstance(current_node, nodes.BatchNorm1DNode):

                        input_size = output_size = current_node.num_features

                        input_scale = current_node.identifier + "_scale"
                        input_bias = current_node.identifier + "_bias"
                        input_mean = current_node.identifier + "_mean"
                        input_var = current_node.identifier + "_var"

                        input_value_info = onnx.helper.make_tensor_value_info(current_input, onnx.TensorProto.FLOAT,
                                                                              [1, input_size])
                        output_value_info = onnx.helper.make_tensor_value_info(current_output, onnx.TensorProto.FLOAT,
                                                                               [1, output_size])
                        scale_value_info = onnx.helper.make_tensor_value_info(input_scale, onnx.TensorProto.FLOAT,
                                                                              [input_size])
                        bias_value_info = onnx.helper.make_tensor_value_info(input_bias, onnx.TensorProto.FLOAT,
                                                                             [input_size])
                        mean_value_info = onnx.helper.make_tensor_value_info(input_mean, onnx.TensorProto.FLOAT,
                                                                             [input_size])
                        var_value_info = onnx.helper.make_tensor_value_info(input_var, onnx.TensorProto.FLOAT,
                                                                            [input_size])

                        scale_tensor = onnx.numpy_helper.from_array(current_node.weight, input_scale)
                        bias_tensor = onnx.numpy_helper.from_array(current_node.bias, input_bias)
                        mean_tensor = onnx.numpy_helper.from_array(current_node.running_mean, input_mean)
                        var_tensor = onnx.numpy_helper.from_array(current_node.running_var, input_var)

                        onnx_node = onnx.helper.make_node(
                            'BatchNormalization',
                            inputs=[current_input, input_scale, input_bias, input_mean, input_var],
                            outputs=[current_output],
                            epsilon=current_node.eps,
                            momentum=current_node.momentum
                        )

                        input_info.append(input_value_info)
                        input_info.append(scale_value_info)
                        input_info.append(bias_value_info)
                        input_info.append(mean_value_info)
                        input_info.append(var_value_info)

                        output_info.append(output_value_info)

                        initializers.append(scale_tensor)
                        initializers.append(bias_tensor)
                        initializers.append(mean_tensor)
                        initializers.append(var_tensor)

                        onnx_nodes.append(onnx_node)

                    else:
                        raise NotImplementedError

                    previous_output = current_output

                onnx_graph = onnx.helper.make_graph(
                    nodes=onnx_nodes,
                    name=network.identifier,
                    inputs=[input_info[0]],
                    outputs=[output_info[-1]],
                    initializer=initializers,
                    value_info=input_info
                )

                onnx_network = onnx.helper.make_model(graph=onnx_graph)
                alt_net = ONNXNetwork(network.identifier + "_onnx", onnx_network)

            else:
                raise NotImplementedError

        return alt_net

    def to_neural_network(self, alt_rep: ONNXNetwork) -> networks.NeuralNetwork:
        """
        Convert the ONNX representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : ONNXNetwork
            The ONNX Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of ONNX Representation.

        """
        identifier = alt_rep.identifier.replace("_onnx", "")
        network = networks.SequentialNetwork(identifier)

        parameters = {}
        for initializer in alt_rep.onnx_network.graph.initializer:
            parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

        shape_info = {}
        for value_info in alt_rep.onnx_network.graph.value_info:
            shape = []
            for dim in value_info.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            shape_info[value_info.name] = shape

        node_index = 1
        for node in alt_rep.onnx_network.graph.node:

            if node.op_type == "Relu":

                # We assume that the real input of the node is always the first element of node.input
                # and that the shape is [batch_placeholder, real_size] for the inputs.
                num_features = shape_info[node.input[0]][1]
                network.add_node(nodes.ReLUNode(f"ReLU_{node_index}", num_features))

            elif node.op_type == "Sigmoid":
                num_features = shape_info[node.input[0]][1]
                network.add_node(nodes.SigmoidNode(f"Sigmoid_{node_index}", num_features))

            elif node.op_type == "Gemm":
                # We assume that the weight tensor is always the second element of node.input and the bias tensor
                # is always the third.
                # N.B: The Marabou procedure for reading ONNX models do not consider the attributes transA and transB,
                # therefore we need to transpose the weight vector.
                weight = parameters[node.input[1]].T
                bias = parameters[node.input[2]]
                in_features = weight.shape[1]
                out_features = weight.shape[0]
                network.add_node(nodes.FullyConnectedNode(f"Linear_{node_index}", in_features,
                                                          out_features, weight, bias))
            elif node.op_type == "BatchNormalization":
                # We assume that the real input is always the first element of node.input, the weight tensor
                # is always the second, the bias tensor is always the third, the running_mean always the fourth
                # and the running_var always the fifth.
                num_features = shape_info[node.input[0]][1]
                weight = parameters[node.input[1]]
                bias = parameters[node.input[2]]
                running_mean = parameters[node.input[3]]
                running_var = parameters[node.input[4]]
                # We assume that eps is always the first attribute and momentum is always the second.
                eps = node.attribute[0].f
                momentum = node.attribute[1].f
                network.add_node(nodes.BatchNorm1DNode(f"BatchNorm_{node_index}", num_features, weight, bias,
                                                       running_mean, running_var, eps, momentum))

            else:
                raise NotImplementedError

            node_index += 1

        return network


class PyTorchConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for PyTorch models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a PyTorchNetwork model.
    to_neural_network(PyTorchNetwork)
        Convert the PyTorchNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> PyTorchNetwork:
        """
        Convert the neural network of interest to a PyTorch representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        PyTorchNetwork
            The PyTorch representation resulting from the conversion of the original network.

        """

        alt_net = None
        pytorch_network = None
        for alt_rep in network.alt_rep_cache:
            if isinstance(alt_rep, PyTorchNetwork) and alt_rep.up_to_date:
                alt_net = alt_rep

        if alt_net is None:

            if not network.up_to_date:

                for alt_rep in network.alt_rep_cache:

                    if alt_rep.up_to_date:

                        if isinstance(alt_rep, ONNXNetwork):
                            onnx_cv = ONNXConverter()
                            network = onnx_cv.to_neural_network(alt_rep)

                        else:
                            raise NotImplementedError
                        break

            if isinstance(network, networks.SequentialNetwork):
                pytorch_layers = []
                for layer in network.nodes.values():

                    new_layer = None
                    if isinstance(layer, nodes.ReLUNode):
                        new_layer = torch.nn.ReLU()

                    elif isinstance(layer, nodes.SigmoidNode):
                        new_layer = torch.nn.Sigmoid()

                    elif isinstance(layer, nodes.FullyConnectedNode):

                        if layer.bias is not None:
                            has_bias = True
                        else:
                            has_bias = False

                        new_layer = torch.nn.Linear(in_features=layer.in_features,
                                                    out_features=layer.out_features,
                                                    bias=has_bias)

                        weight = torch.from_numpy(layer.weight)
                        new_layer.weight.data = weight

                        if has_bias:
                            bias = torch.from_numpy(layer.bias)
                            new_layer.bias.data = bias

                    elif isinstance(layer, nodes.BatchNorm1DNode):

                        new_layer = torch.nn.BatchNorm1d(num_features=layer.num_features,
                                                         eps=layer.eps, momentum=layer.momentum,
                                                         affine=layer.affine,
                                                         track_running_stats=layer.track_running_stats)

                        new_layer.weight.data = torch.from_numpy(layer.weight)
                        new_layer.bias.data = torch.from_numpy(layer.bias)
                        new_layer.running_mean.data = torch.from_numpy(layer.running_mean)
                        new_layer.running_var.data = torch.from_numpy(layer.running_var)

                    if new_layer is not None:
                        pytorch_layers.append(new_layer)

                pytorch_network = torch.nn.Sequential(*pytorch_layers)

            if alt_net is None and pytorch_network is None:
                print("WARNING: network to convert is not valid, the alternative representation is None")

            identifier = network.identifier + '_pytorch'
            alt_net = PyTorchNetwork(identifier=identifier, pytorch_network=pytorch_network)

        return alt_net

    def to_neural_network(self, alt_rep: PyTorchNetwork) -> networks.NeuralNetwork:
        """
        Convert the PyTorch representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : PyTorchNetwork
            The PyTorch Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of PyTorch Representation.

        """

        identifier = alt_rep.identifier.replace('_pytorch', '')
        network = networks.SequentialNetwork(identifier=identifier)

        node_index = 0
        size_prev_output = 0
        alt_rep.pytorch_network.cpu()
        for m in alt_rep.pytorch_network.modules():

            new_node = None

            if isinstance(m, torch.nn.ReLU):
                new_node = nodes.ReLUNode(identifier='ReLU_{}'.format(node_index), num_features=size_prev_output)

            elif isinstance(m, torch.nn.Sigmoid):
                new_node = nodes.SigmoidNode(identifier='Sigmoid_{}'.format(node_index), num_features=size_prev_output)

            elif isinstance(m, torch.nn.Linear):
                in_features = m.in_features
                out_features = m.out_features
                weight = m.weight.detach().numpy()
                bias = m.bias.detach().numpy()
                new_node = nodes.FullyConnectedNode(identifier='FullyConnected_{}'.format(node_index),
                                                    in_features=in_features, out_features=out_features,
                                                    weight=weight, bias=bias)

                size_prev_output = out_features

            elif isinstance(m, torch.nn.BatchNorm1d):

                num_features = m.num_features
                eps = m.eps
                momentum = m.momentum
                track_running_stats = m.track_running_stats
                affine = m.affine

                weight = m.weight.detach().numpy()
                bias = m.bias.detach().numpy()
                running_mean = m.running_mean.numpy()
                running_var = m.running_var.numpy()

                new_node = nodes.BatchNorm1DNode(identifier='BatchNorm1D_{}'.format(node_index),
                                                 num_features=num_features, weight=weight, bias=bias,
                                                 running_mean=running_mean, running_var=running_var, eps=eps,
                                                 momentum=momentum, affine=affine,
                                                 track_running_stats=track_running_stats)

                size_prev_output = num_features

            if new_node is not None:
                node_index += 1
                network.add_node(new_node)

        return network


class TensorflowConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for Tensorflow models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a TensorflowNetwork model.
    to_neural_network(ONNXNetwork)
        Convert the TensorflowNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> TensorflowNetwork:
        """
        Convert the neural network of interest to a Tensorflow representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        TensorflowNetwork
            The Tensorflow representation resulting from the conversion of the original network.

        """

    def to_neural_network(self, alt_rep: TensorflowNetwork) -> networks.NeuralNetwork:
        """
        Convert the Tensorflow representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : TensorflowNetwork
            The Tensorflow Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of Tensorflow Representation.

        """
