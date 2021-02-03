import abc
from pynever.tensor import Tensor
import pynever.networks as networks
from typing import List, Optional
import pynever.strategies.abstraction as abst
import pynever.nodes as nodes
import time
import numpy as np


class Property(abc.ABC):
    """
    An abstract class used to represent a generic property for a NeuralNetwork.
    """


class SMTLIBProperty(Property):
    """
    A concrete class used to represent a generic property for a NeuralNetwork expressed as a SMTLIB query.

    Attributes
    ----------
    smtlib_path : str
        Filepath for the SMTLIB file in which the property is defined.

    """

    def __init__(self, smtlib_path: str):
        self.smtlib_path = smtlib_path


class LocalRobustnessProperty(Property):
    """
    A concrete class used to represent a local robustness property for a NeuralNetwork.
    Formally the property check if the counterexample (i.e., the adversarial example) exists, therefore
    when the verification strategy check such property it should return True if the adversarial example exist and
    false otherwise.

    Attributes
    ----------
    data : Tensor
        Original data used to determine the local robustness.
    target : int
        If targeted is True then it is the desired target for the adversarial, otherwise it is the correct target of
        data.
    targeted : bool
        Flag which is True if the robustness property is targeted, False otherwise.
    norm : str
        Norm type used to determine the local robustness. At present the only acceptable value is Linf.
    epsilon : float
        Magnitude of the acceptable perturbation.
    bounds: list
        List of (lower_bound, upper_bound) for the data.

    """

    def __init__(self, data: Tensor, target: int, targeted: bool, norm: str, epsilon: float, bounds: list):

        self.data = data
        self.target = target
        self.targeted = targeted
        if norm != "Linf":
            raise NotImplementedError
        self.norm = norm
        self.epsilon = epsilon
        self.bounds = bounds


class NeVerProperty(Property):
    """
    A concrete class used to represent a NeVer property for a NeuralNetwork. We assume that the hyperplane
    out_coef_mat * y <= out_bias_mat represent the unsafe region (i.e., the negation of the desired property).
    At present the input set must be defined as in_coef_mat * x <= in_bias_mat

    Attributes
    ----------
    in_coef_mat: Tensor
        Matrix of the coefficients for the input constraints.
    in_bias_mat: Tensor
        Matrix of the biases for the input constraints.
    out_coef_mat: List[Tensor]
        Matrixes of the coefficients for the output constraints.
    out_bias_mat: List[Tensor]
        Matrixes of the biases for the output constraints.

    """

    def __init__(self, in_coef_mat: Tensor, in_bias_mat: Tensor, out_coef_mat: List[Tensor],
                 out_bias_mat: List[Tensor]):

        self.in_coef_mat = in_coef_mat
        self.in_bias_mat = in_bias_mat
        self.out_coef_mat = out_coef_mat
        self.out_bias_mat = out_bias_mat


class VerificationStrategy(abc.ABC):
    """
    An abstract class used to represent a Verification Strategy.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        bool
            True is the neural network satisfy the property, False otherwise.

        """
        pass


class NeverVerification(VerificationStrategy):

    """
    Class used to represent the Never verification strategy.

    Attributes
    ----------
    log_filepath: str
        Filepath for saving the log files of the verification procedure.

    neuron_relevance : bool
        Flag to determine the heuristic used to decide how to refine the abstraction. If True the refinement
        decision is taken at the neuron level. Otherwise it is taken on the single sub stars. See the class
        abstraction.AbsReLUNode for details.

    iqr_thresholding : bool
        Flag used to determine if use the iqr thresholding heuristic in the refinement

    iqr_mult : float
        Multiplier for the iqr value.

    refinement_percentage: float
        Percentage of the neurons (if neuron_relevance = True) or the stars (if neuron_relevance = False) which
        will use the complete version of the abstraction algorithm (instead of the overapproximate version).

    refinement_level: int
        Refinement level for the sigmoid abstraction.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument.
    """

    def __init__(self, log_filepath: str, neuron_relevance: bool, iqr_thresholding: bool,
                 iqr_mult: float, refinement_percentage: float, refinement_level: int):

        assert 0 <= refinement_percentage <= 1, "refinement_percentage must be a value between 0 and 1"

        self.log_filepath = log_filepath
        self.neuron_relevance = neuron_relevance
        self.refinement_percentage = refinement_percentage
        self.iqr_thresholding = iqr_thresholding
        self.iqr_mult = iqr_mult
        self.refinement_level = refinement_level

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):

        assert isinstance(network, networks.SequentialNetwork), "Only sequential networks are supported at present"
        abst_networks = abst.AbsSeqNetwork("Abstract Network")

        current_node = network.get_first_node()
        while current_node is not None:

            if isinstance(current_node, nodes.FullyConnectedNode):
                abst_networks.add_node(abst.AbsFullyConnectedNode("ABST_" + current_node.identifier, current_node))

            elif isinstance(current_node, nodes.ReLUNode):
                abst_networks.add_node(abst.AbsReLUNode("ABST_" + current_node.identifier, current_node,
                                                        self.neuron_relevance, self.iqr_thresholding, self.iqr_mult,
                                                        self.refinement_percentage))

            elif isinstance(current_node, nodes.SigmoidNode):
                abst_networks.add_node(abst.AbsSigmoidNode("ABST_" + current_node.identifier, current_node,
                                                           self.refinement_level))

            else:
                raise NotImplementedError

            current_node = network.get_next_node(current_node)

        with open(self.log_filepath, "w") as log_file:
            areas_log = self.log_filepath.replace(".txt", "_areas.txt")
            with open(areas_log, "w") as area_log_file:

                if isinstance(prop, NeVerProperty):

                    input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
                    input_starset = abst.StarSet({input_star})
                    current_node = abst_networks.get_first_node()
                    output_starset = input_starset
                    while current_node is not None:

                        time_start = time.perf_counter()
                        output_starset = current_node.forward(output_starset)
                        time_end = time.perf_counter()

                        print(f"Computing starset for layer {current_node.identifier}. Current starset has dimension "
                              f"{len(output_starset.stars)}. Time to compute: {time_end - time_start}s.")
                        log_file.write(f"Computing starset for layer {current_node.identifier}. "
                                       f"Current starset has dimension {len(output_starset.stars)}."
                                       f"Time to compute: {time_end - time_start}s.\n")

                        if isinstance(current_node, abst.AbsReLUNode):
                            for areas in current_node.overapprox_areas:
                                if isinstance(areas, np.ndarray):
                                    area_log_file.write(f"{areas.tolist()}\n")
                                else:
                                    area_log_file.write(f"{areas}\n")

                        current_node = abst_networks.get_next_node(current_node)

                    out_coef_mat = prop.out_coef_mat
                    out_bias_mat = prop.out_bias_mat

                else:
                    raise NotImplementedError

                verified = True
                for i in range(len(out_coef_mat)):

                    for star in output_starset.stars:
                        out_coef = out_coef_mat[i]
                        out_bias = out_bias_mat[i]
                        temp_star = abst.intersect_with_halfspace(star, out_coef, out_bias)
                        if not temp_star.check_if_empty():
                            verified = False
                            # print(f"Star {k}: Unsafe")

                log_file.write(f"Verification Result: {verified}.\n")

        return verified
