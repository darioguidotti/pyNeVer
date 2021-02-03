import abc
from pynever.tensor import Tensor
import numpy as np


class LayerNode(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Layer of a Neural Network.
    Its concrete children correspond to real network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    """

    def __init__(self, identifier: str):

        self.identifier = identifier


class ReLUNode(LayerNode):
    """
    A class used for our internal representation of a ReLU Layer of a Neural Network.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.
    num_features : int
        Number of input and output features of the ReLU layer.
    """

    def __init__(self, identifier: str, num_features):
        super().__init__(identifier)
        self.num_features = num_features

    def __repr__(self):

        return f"ReLUNode - num_features = {self.num_features}"

    def to_string(self):

        return f"ReLUNode - num_features = {self.num_features}"


class SigmoidNode(LayerNode):
    """
    A class used for our internal representation of a Sigmoid Layer of a Neural Network.
    TO DO: It is not supported in the existing conversion strategies.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.
    num_features : int
        Number of input and output features of the Sigmoid layer.

    """

    def __init__(self, identifier: str, num_features):
        super().__init__(identifier)
        self.num_features = num_features

    def __repr__(self):

        return f"SigmoidNode - num_features = {self.num_features}"

    def to_string(self):

        return f"SigmoidNode - num_features = {self.num_features}"


class FullyConnectedNode(LayerNode):
    """
    A class used for our internal representation of a Fully Connected layer of a Neural Network

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.
    in_features : int
        Number of input features of the fully connected layer.
    out_features : int
        Number of output features of the fully connected layer.
    weight : Tensor, optional
        Tensor containing the weight parameters of the fully connected layer.
    bias : Tensor, optional
        Tensor containing the bias parameters of the fully connected layer.
    """

    def __init__(self, identifier: str, in_features: int, out_features: int,
                 weight: Tensor = None, bias: Tensor = None):

        super().__init__(identifier)
        self.in_features = in_features
        self.out_features = out_features

        if weight is None:
            weight = np.random.normal(size=[out_features, in_features])

        if bias is None:
            bias = np.random.normal(size=[out_features])

        self.weight = weight
        self.bias = bias

    def __repr__(self):

        return f"FullyConnectedNode - in_features = {self.in_features}, out_features = {self.out_features}"

    def to_string(self):

        return f"FullyConnectedNode - in_features = {self.in_features}, out_features = {self.out_features}"


class BatchNorm1DNode(LayerNode):
    """
    A class used for our internal representation of a one dimensional Batch Normalization Layer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.
    num_features : int
        Number of input and output feature of the Batch Normalization Layer.
    weight : Tensor, optional
        Tensor containing the weight parameters of the Batch Normalization Layer.
    bias : Tensor, optional
        Tensor containing the bias parameter of the Batch Normalization Layer.
    running_mean : Tensor, optional
        Tensor containing the running mean parameter of the Batch Normalization Layer.
    running_var : Tensor, optional
        Tensor containing the running var parameter of the Batch Normalization Layer.
    eps : float, optional
        Value added to the denominator for numerical stability (default: 1e-5).
    momentum : float, optional
        Value used for the running_mean and running_var computation. Can be set to None
        for cumulative moving average (default: 0.1)
    affine : bool, optional
        When set to True, the module has learnable affine parameter (default: True).
    track_running_stats : bool, optional
        When set to True, the module tracks the running mean and variance, when set to false the module
        does not track such statistics and always uses batch statistics in both training and eval modes (default: True).

    """

    def __init__(self, identifier: str, num_features: int, weight: Tensor = None, bias: Tensor = None,
                 running_mean: Tensor = None, running_var: Tensor = None, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):

        super().__init__(identifier)
        self.num_features = num_features

        if track_running_stats and running_mean is None and running_var is None:
            running_mean = np.ones(num_features)
            running_var = np.zeros(num_features)

        if weight is None:
            weight = np.ones(num_features)

        if bias is None:
            bias = np.zeros(num_features)

        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def __repr__(self):

        return f"BatchNorm1DNode - num_features = {self.num_features}"

    def to_string(self):
        return f"BatchNorm1DNode - num_features = {self.num_features}"
