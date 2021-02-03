import abc
import pynever.nodes as nodes
import uuid
import numpy as np
import multiprocessing
import itertools
from typing import Set, List, Union
import math
import scipy.stats as sts

from pynever.tensor import Tensor
from ortools.linear_solver import pywraplp


# abstraction_logfile = "logs/ABSTRACTION_LOG.txt"


class AbsElement(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Abstract Element (e.g., interval, zonotope,
    polyhedra etc.)

    Attributes
    ----------
    identifier : str
        Identifier of the AbsElement.
    """

    def __init__(self, identifier: str = None):
        if identifier is None:
            self.identifier = uuid.uuid4()
        else:
            self.identifier = identifier


class Star:
    """
    A concrete class used for our internal representation of a Star.
    The Star is defined as {x | x = c + Va such that Ca <= d}
    where c is a n-dimensional vector corresponding to the center of the Star.
    V is the n-by-m matrix composed by the basis vectors.
    a is the vector of m variables, C (p-by-m) and d (p-dim) are the matrix and the biases
    defining a set of constraints.

    We refer to <Star-Based Reachability Analysis of Deep Neural Networks>
    (https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39) for details.

    Attributes
    ----------
    center : Tensor
        Center of the Star.
    basis_matrix : Tensor
        Matrix composed by the basis vectors of the Star
    predicate_matrix : Tensor
        Matrix of the Predicate.
    predicate_bias : Tensor
        Bias of the Predicate.
    ubs : list
        Upper bounds of the points defined by the Star.
    lbs : list
        Lower bounds of the points defined by the Star.
    is_empty : bool
        Boolean flag: True if the Star defines an empty set of points, False otherwise

    Methods
    ----------
    get_bounds()
        Function used to get the the upper and lower bounds of the n variables of the star.

    """

    def __init__(self, predicate_matrix: Tensor, predicate_bias: Tensor, center: Tensor = None,
                 basis_matrix: Tensor = None, lbs: list = None, ubs: list = None, is_empty: bool = None):

        predicate_dim_message = f"Error: the first dimension of the predicate_matrix ({predicate_matrix.shape[0]}) " \
                                f"must be equal to the dimension of the predicate_bias ({predicate_bias.shape[0]})."
        assert predicate_matrix.shape[0] == predicate_bias.shape[0], predicate_dim_message

        self.predicate_matrix = predicate_matrix
        self.predicate_bias = predicate_bias

        if center is None and basis_matrix is None:
            self.center = np.zeros((predicate_matrix.shape[1], 1))
            self.basis_matrix = np.identity(predicate_matrix.shape[1])

        else:
            center_dim_message = f"Error: the first dimension of the basis_matrix ({basis_matrix.shape[0]}) " \
                                 f"must be equal to the dimension of the center ({center.shape[0]})."
            assert center.shape[0] == basis_matrix.shape[0], center_dim_message

            basis_dim_message = f"Error: the second dimension of the basis_matrix ({basis_matrix.shape[1]}) " \
                                f"must be equal to the second dimension of the predicate_matrix " \
                                f"({predicate_matrix.shape[1]})."
            assert basis_matrix.shape[1] == predicate_matrix.shape[1], basis_dim_message

            self.center = center
            self.basis_matrix = basis_matrix

        if lbs is None:
            lbs = [None for i in range(self.center.shape[0])]

        if ubs is None:
            ubs = [None for i in range(self.center.shape[0])]

        self.lbs = lbs
        self.ubs = ubs
        self.is_empty = is_empty

    def check_if_empty(self) -> bool:
        """
        Function used to check if the set of points defined by the star is empty.

        Return
        ---------
        bool
            True if the star defines an empty set of points, False otherwise.
        """
        #start_time = time.perf_counter()
        if self.is_empty is None:

            solver, alphas, constraints = self.__get_predicate_lp_solver()
            objective = solver.Objective()
            for j in range(self.predicate_matrix.shape[1]):
                objective.SetCoefficient(alphas[j], 0)
            objective.SetOffset(0)

            objective.SetMinimization()
            status = solver.Solve()
            if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
                self.is_empty = True
            else:
                self.is_empty = False

        #end_time = time.perf_counter()
        #with open(abstraction_logfile, "a") as f:
        #    f.write(f"Empty,{end_time - start_time}\n")

        return self.is_empty

    def get_bounds(self, i) -> (float, float):
        """
        Function used to get the the upper and lower bounds of the n variables of the star.

        Return
        ---------
        (float, float)
            Tuple containing the lower and upper bounds of the variable i of the star
        """
        #start_time = time.perf_counter()
        if self.lbs[i] is None or self.ubs[i] is None:

            solver, alphas, constraints = self.__get_predicate_lp_solver()
            objective = solver.Objective()
            for j in range(self.basis_matrix.shape[1]):
                objective.SetCoefficient(alphas[j], self.basis_matrix[i, j])
            objective.SetOffset(self.center[i, 0])

            objective.SetMinimization()

            status = solver.Solve()

            """if status == pywraplp.Solver.ABNORMAL:
                objective = solver.Objective()
                for j in range(self.basis_matrix.shape[1]):
                    objective.SetCoefficient(alphas[j], 1)
                objective.SetOffset(0)
                objective.SetMinimization()
                status = solver.Solve()
                value = solver.Objective().Value()
                print(status)"""

            assert status == pywraplp.Solver.OPTIMAL, "The LP problem was not Optimal"

            lb = solver.Objective().Value()

            objective.SetMaximization()
            status = solver.Solve()
            #temp = solver.Objective().Value()
            assert status == pywraplp.Solver.OPTIMAL, "The LP problem was not Optimal"

            ub = solver.Objective().Value()

            self.lbs[i] = lb
            self.ubs[i] = ub

        else:
            lb = self.lbs[i]
            ub = self.ubs[i]

        #end_time = time.perf_counter()
        #with open(abstraction_logfile, "a") as f:
        #    f.write(f"Bound,{end_time - start_time}\n")
        return lb, ub

    def __get_predicate_lp_solver(self) -> (pywraplp.Solver, list, list):
        """
        Creates an lp solver using pywraplp and adds the variables and constraints
        corresponding to the predicate of the star.

        Returns
        ---------
        (pywraplp.Solver, list, list)
            Respectively the lp solver, the variables and the constraints.
        """

        solver = pywraplp.Solver.CreateSolver('GLOP')
        alphas = []
        for j in range(self.basis_matrix.shape[1]):
            new_alpha = solver.NumVar(-solver.infinity(), solver.infinity(), f'alpha_{j}')
            alphas.append(new_alpha)

        constraints = []
        for k in range(self.predicate_matrix.shape[0]):
            new_constraint = solver.Constraint(-solver.infinity(), self.predicate_bias[k, 0])
            for j in range(self.predicate_matrix.shape[1]):
                new_constraint.SetCoefficient(alphas[j], self.predicate_matrix[k, j])
            constraints.append(new_constraint)

        return solver, alphas, constraints


class StarSet(AbsElement):
    """
    Concrete class for our internal representation of a StarSet abstract element. A StarSet consist in a set
    of Star objects.

    Attributes
    ----------
    stars : Set[Star]
        Set of Star objects.
    """

    def __init__(self, stars: Set[Star] = None, identifier: str = None):
        super().__init__(identifier)
        if stars is None:
            self.stars = set()
        else:
            self.stars = stars


def intersect_with_halfspace(star: Star, coef_mat: Tensor, bias_mat: Tensor) -> Star:

    new_center = star.center
    new_basis_matrix = star.basis_matrix
    hs_pred_matrix = np.matmul(coef_mat, star.basis_matrix)
    hs_pred_bias = bias_mat - np.matmul(coef_mat, star.center)
    new_pred_matrix = np.vstack((star.predicate_matrix, hs_pred_matrix))
    new_pred_bias = np.vstack((star.predicate_bias, hs_pred_bias))

    new_star = Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix)
    return new_star


def __mixed_step_relu(abs_input: Set[Star], var_index: int, refinement_flag: bool,
                      refinement_percentage: float = None) -> Set[Star]:

    abs_input = list(abs_input)
    abs_output = set()

    # Compute areas used to decide if the refinement is necessary.
    bounds = []
    t_areas = []
    for i in range(len(abs_input)):
        if abs_input[i].check_if_empty():
            bounds.append((None, None))
            t_areas.append(0)
        else:
            temp_lb, temp_ub = abs_input[i].get_bounds(var_index)
            bounds.append((temp_lb, temp_ub))
            if temp_lb < 0 and temp_ub > 0:
                t_areas.append(-temp_lb * temp_ub / 2.0)
            else:
                t_areas.append(0)

    t_areas = np.array(t_areas)

    # If the refinement_percentage is not None and the neuron should be refined I use the percentage
    # heuristic to select which stars I want to refine.
    if len(abs_input) == 1 and refinement_flag and refinement_percentage is not None and refinement_percentage > 0:
        ref_flags = [True]
    elif refinement_percentage is not None and refinement_flag:

        sorted_indexes = np.flip(np.argsort(t_areas))
        sorted_areas = t_areas[sorted_indexes]

        num_non_zero = 0
        for i in range(len(sorted_areas)):
            if sorted_areas[i] != 0:
                num_non_zero = num_non_zero + 1

        num_to_refine = int(refinement_percentage * num_non_zero)

        i = 0
        index_star_to_refine = []
        while i < num_to_refine:
            index_star_to_refine.append(sorted_indexes[i])
            i = i + 1

        ref_flags = []
        for i in range(len(t_areas)):
            if i in index_star_to_refine:
                ref_flags.append(True)
            else:
                ref_flags.append(False)

    # Otherwise I chose to refine or not refine them only based on the neuron refinement flag.
    else:
        ref_flags = [refinement_flag for i in range(len(abs_input))]

    for i in range(len(abs_input)):

        star = abs_input[i]
        ref_flag = ref_flags[i]

        if not star.check_if_empty():

            lb, ub = bounds[i]
            mask = np.identity(star.center.shape[0])
            mask[var_index, var_index] = 0

            if lb >= 0:
                abs_output = abs_output.union({star})

            elif ub <= 0:
                new_center = np.matmul(mask, star.center)
                new_basis_mat = np.matmul(mask, star.basis_matrix)
                new_pred_mat = star.predicate_matrix
                new_pred_bias = star.predicate_bias
                new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)
                abs_output = abs_output.union({new_star})

            else:

                if ref_flag:

                    # Creating lower bound star.
                    lower_star_center = np.matmul(mask, star.center)
                    lower_star_basis_mat = np.matmul(mask, star.basis_matrix)
                    # Adding x <= 0 constraints to the predicate.
                    lower_predicate_matrix = np.vstack((star.predicate_matrix, star.basis_matrix[var_index, :]))
                    # Possibile problema sulla dimensionalita' di star.center[var_index]
                    lower_predicate_bias = np.vstack((star.predicate_bias, -star.center[var_index]))
                    lower_star = Star(lower_predicate_matrix, lower_predicate_bias, lower_star_center,
                                      lower_star_basis_mat)

                    # Creating upper bound star.
                    upper_star_center = star.center
                    upper_star_basis_mat = star.basis_matrix
                    # Adding x >= 0 constraints to the predicate.
                    upper_predicate_matrix = np.vstack((star.predicate_matrix, -star.basis_matrix[var_index, :]))
                    # Possibile problema sulla dimensionalita' di star.center[var_index]
                    upper_predicate_bias = np.vstack((star.predicate_bias, star.center[var_index]))
                    upper_star = Star(upper_predicate_matrix, upper_predicate_bias, upper_star_center,
                                      upper_star_basis_mat)

                    abs_output = abs_output.union({lower_star, upper_star})

                else:

                    col_c_mat = star.predicate_matrix.shape[1]
                    row_c_mat = star.predicate_matrix.shape[0]

                    c_mat_1 = np.zeros((1, col_c_mat + 1))
                    c_mat_1[0, col_c_mat] = -1
                    c_mat_2 = np.hstack((np.array([star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
                    coef_3 = - ub / (ub - lb)
                    c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
                    c_mat_0 = np.hstack((star.predicate_matrix, np.zeros((row_c_mat, 1))))

                    d_0 = star.predicate_bias
                    d_1 = np.zeros((1, 1))
                    d_2 = -star.center[var_index] * np.ones((1, 1))
                    d_3 = np.array([(ub / (ub - lb)) * (star.center[var_index] - lb)])

                    new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3))
                    new_pred_bias = np.vstack((d_0, d_1, d_2, d_3))

                    new_center = np.matmul(mask, star.center)
                    temp_basis_mat = np.matmul(mask, star.basis_matrix)
                    temp_vec = np.zeros((star.basis_matrix.shape[0], 1))
                    temp_vec[var_index, 0] = 1
                    new_basis_mat = np.hstack((temp_basis_mat, temp_vec))

                    new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

                    abs_output = abs_output.union({new_star})

    return abs_output


def mixed_single_relu_forward(star: Star, neuron_relevance: bool, iqr_thresholding: bool, iqr_mult: float,
                              refinement_percentage: float) -> (Set[Star], Tensor):

    temp_abs_input = {star}
    if star.check_if_empty():
        return set()
    else:

        # If neuron_relevance is True it means that we want to chose which neurons will use the complete relu at this
        # level and not for the sub-stars generated during the computation of the various output neurons. Moreover it
        # means that the refinement_percentage is in relation with the number of neurons, not the number of sub-stars.
        n_areas = []
        if neuron_relevance:

            for i in range(star.center.shape[0]):
                lb, ub = star.get_bounds(i)
                if lb < 0 and ub > 0:
                    n_areas.append(-lb * ub / 2.0)
                else:
                    n_areas.append(0)

            n_areas = np.array(n_areas)

            sorted_indexes = np.flip(np.argsort(n_areas))
            sorted_areas = n_areas[sorted_indexes]

            num_non_zero = 0
            for i in range(len(sorted_areas)):
                if sorted_areas[i] > 0:
                    num_non_zero = num_non_zero + 1

            num_to_refine = int(refinement_percentage * num_non_zero)

            i = 0
            index_star_to_refine = []
            while i < num_to_refine:
                index_star_to_refine.append(sorted_indexes[i])
                i = i + 1

            refinement_flags = []
            for i in range(len(n_areas)):
                if i in index_star_to_refine:
                    refinement_flags.append(True)
                else:
                    refinement_flags.append(False)

            refinement_percentage = None

        elif iqr_thresholding:

            for i in range(star.center.shape[0]):
                lb, ub = star.get_bounds(i)
                if lb < 0 and ub > 0:
                    n_areas.append(-lb * ub / 2.0)
                else:
                    n_areas.append(0)

            n_areas = np.array(n_areas)

            threshold = np.median(n_areas) + iqr_mult * sts.iqr(n_areas)
            refinement_flags = []
            for i in range(len(n_areas)):
                if n_areas[i] > threshold:
                    refinement_flags.append(True)
                else:
                    refinement_flags.append(False)

            refinement_percentage = None

        # Otherwise the decision is left to the __mixed_step_relu function and the refinement_percentage is referred
        # to the number of sub-star. The refinement flags are set to True since potentially every neurons can be
        # refined.
        else:
            refinement_flags = [True for i in range(star.center.shape[0])]

        for i in range(star.center.shape[0]):
            temp_abs_input = __mixed_step_relu(temp_abs_input, i, refinement_flags[i], refinement_percentage)

        return temp_abs_input, n_areas


def single_fc_forward(star: Star, weight: Tensor, bias: Tensor) -> Set[Star]:
    """
    Utility function for the management of the forward for AbsFullyConnectedNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.
    """
    assert (weight.shape[1] == star.basis_matrix.shape[0])

    new_basis_matrix = np.matmul(weight, star.basis_matrix)
    new_center = np.matmul(weight, star.center) + bias
    new_predicate_matrix = star.predicate_matrix
    new_predicate_bias = star.predicate_bias

    new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix)

    return {new_star}


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sig_fod(x: float) -> float:
    return math.exp(-x) / math.pow(1 + math.exp(-x), 2)


def area_sig_triangle(lb: float, ub: float) -> float:

    x_p = (ub * sig_fod(ub) - lb * sig_fod(lb)) / (sig_fod(ub) - sig_fod(lb)) - \
          (sig(ub) - sig(lb)) / (sig_fod(ub) - sig_fod(lb))

    y_p = sig_fod(ub) * (x_p - ub) + sig(ub)

    height = abs(y_p - (sig(ub) - sig(lb)) / (ub - lb) * x_p + sig(lb) - lb * (sig(ub) - sig(lb)) / (ub - lb)) / \
             math.sqrt(1 + math.pow((sig(ub) - sig(lb)) / (ub - lb), 2))

    base = math.sqrt(math.pow(ub - lb, 2) + math.pow(sig(ub) - sig(lb), 2))

    return base * height / 2.0


def __recursive_step_sigmoid(star: Star, var_index: int, approx_level: int, lb: float, ub: float, tolerance: float) -> Set[Star]:

    assert approx_level >= 0

    if abs(ub - lb) < tolerance:

        if ub <= 0:
            if ub + tolerance > 0:
                ub = 0
            else:
                ub = ub + tolerance
            lb = lb - tolerance
        else:
            if lb - tolerance < 0:
                lb = 0
            else:
                lb = lb - tolerance
            ub = ub + tolerance

    assert (lb <= 0 and ub <= 0) or (lb >= 0 and ub >= 0)

    mask = np.identity(star.center.shape[0])
    mask[var_index, var_index] = 0

    if approx_level == 0:

        if lb < 0 and ub <= 0:

            c_mat_1 = np.hstack((np.array([sig_fod(lb) * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
            c_mat_2 = np.hstack((np.array([sig_fod(ub) * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
            coef_3 = - (sig(ub) - sig(lb)) / (ub - lb)
            c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), np.ones((1, 1))))

            d_1 = np.array([-sig_fod(lb) * (star.center[var_index] - lb) - sig(lb)])
            d_2 = np.array([-sig_fod(ub) * (star.center[var_index] - ub) - sig(ub)])
            d_3 = np.array([-coef_3 * (star.center[var_index] - lb) + sig(lb)])

        else:

            c_mat_1 = np.hstack((np.array([-sig_fod(lb) * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
            c_mat_2 = np.hstack((np.array([-sig_fod(ub) * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
            coef_3 = (sig(ub) - sig(lb)) / (ub - lb)
            c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))

            d_1 = np.array([sig_fod(lb) * (star.center[var_index] - lb) + sig(lb)])
            d_2 = np.array([sig_fod(ub) * (star.center[var_index] - ub) + sig(ub)])
            d_3 = np.array([-coef_3 * (star.center[var_index] - lb) - sig(lb)])

        col_c_mat = star.predicate_matrix.shape[1]

        # Adding lb and ub bounds to enhance stability
        c_mat_lb = np.zeros((1, col_c_mat + 1))
        c_mat_lb[0, col_c_mat] = -1
        d_lb = -sig(lb) * np.ones((1, 1))

        c_mat_ub = np.zeros((1, col_c_mat + 1))
        c_mat_ub[0, col_c_mat] = 1
        d_ub = sig(ub) * np.ones((1, 1))

        row_c_mat = star.predicate_matrix.shape[0]
        c_mat_0 = np.hstack((star.predicate_matrix, np.zeros((row_c_mat, 1))))
        d_0 = star.predicate_bias

        new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3, c_mat_lb, c_mat_ub))
        new_pred_bias = np.vstack((d_0, d_1, d_2, d_3, d_lb, d_ub))

        new_center = np.matmul(mask, star.center)
        temp_basis_mat = np.matmul(mask, star.basis_matrix)
        temp_vec = np.zeros((star.basis_matrix.shape[0], 1))
        temp_vec[var_index, 0] = 1
        new_basis_mat = np.hstack((temp_basis_mat, temp_vec))

        new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

        return {new_star}

    else:

        # We need to select the boundary between lb and ub. The optimal boundary is the one which minimizes the
        # area of the two resulting triangle. Since computing the optimal is too slow we do an approximate search
        # between lb and ub considering s search points.

        num_search_points = 10
        boundaries = np.linspace(lb, ub, num_search_points, endpoint=False)
        boundaries = boundaries[1:]

        best_boundary = None
        smallest_area = 99999999
        for boundary in boundaries:
            area_1 = area_sig_triangle(lb, boundary)
            area_2 = area_sig_triangle(boundary, ub)
            if area_1 + area_2 < smallest_area:
                smallest_area = area_1 + area_2
                best_boundary = boundary

        star_set = set()
        star_set = star_set.union(__recursive_step_sigmoid(star, var_index, approx_level - 1, lb, best_boundary, tolerance))
        star_set = star_set.union(__recursive_step_sigmoid(star, var_index, approx_level - 1, best_boundary, ub, tolerance))

        return star_set


def __approx_step_sigmoid(abs_input: Set[Star], var_index: int, approx_level: int, tolerance: float) -> Set[Star]:

    abs_output = set()
    for star in abs_input:

        if not star.check_if_empty():
            lb, ub = star.get_bounds(var_index)

            if (lb < 0) and (ub > 0):
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, lb, 0,
                                                                       tolerance))
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, 0, ub,
                                                                       tolerance))
            else:
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, lb,
                                                                       ub, tolerance))

    return abs_output


def single_sigmoid_forward(star: Star, approx_levels: List[int]) -> Set[Star]:
    tolerance = 0.01
    temp_abs_input = {star}
    for i in range(star.center.shape[0]):
        temp_abs_input = __approx_step_sigmoid(temp_abs_input, i, approx_levels[i], tolerance)
        print(f"Index {i}, NumStar: {len(temp_abs_input)}")
    return temp_abs_input


class RefinementState(abc.ABC):
    """
    A class used for the internal control of the refinement strategies/heuristics applied in the abstraction refinement
    step.
    """
    pass


class AbsLayerNode(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Abstract Transformer Layer of an
    AbsNeural Network. Its concrete children correspond to real abstract interpretation network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the AbsLayerNode.

    ref_node : LayerNode
        LayerNode di riferimento per l'abstract transformer.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction.

    """

    def __init__(self, identifier: str, ref_node: nodes.LayerNode):
        self.identifier = identifier
        self.ref_node = ref_node

    @abc.abstractmethod
    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        pass

    @abc.abstractmethod
    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsReLUNode(AbsLayerNode):
    """
    A class used for our internal representation of a ReLU Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : ReLUNode
        LayerNode di riferimento per l'abstract transformer.

    neuron_relevance : bool, optional
        Flag used to determine if the refinement should be done with relation to the neurons or the stars
        (default: True).

    iqr_thresholding : bool, optional
        Flag used to determine if the refinement should be done using the interquartile range of the areas of the
        overapproximation as a threshold.

    iqr_mult : float
        Multiplier for the iqr value.

    refinement_percentage : float, optional
        Percentage of the neurons (or stars) we chose to refine (default: 0).

    overapprox_areas : Tensor
        Tensor containing the areas of the overapproximation of the star passed in the forward with respect to the
        different neurons of the layer. It must have dimension number of stars x number of neurons.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction.
    """

    def __init__(self, identifier: str, ref_node: nodes.ReLUNode, neuron_relevance: bool = True,
                 iqr_thresholding: bool = True, iqr_mult: float = 1, refinement_percentage: float = 0):

        super().__init__(identifier, ref_node)
        assert 0 <= refinement_percentage <= 1, "refinement_percentage must be a value between 0 and 1"

        self.neuron_relevance = neuron_relevance
        self.refinement_percentage = refinement_percentage
        self.overapprox_areas = None
        self.iqr_thresholding = iqr_thresholding
        self.iqr_mult = iqr_mult

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        # parallel is used for debug: it permits to return to a sequential procedure
        parallel = True
        if isinstance(abs_input, StarSet):
            if parallel:
                return self.__parallel_starset_forward(abs_input)
            else:
                return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass

    def __parallel_starset_forward(self, abs_input: StarSet) -> StarSet:

        #with open(abstraction_logfile, "a") as f:
        #    f.write("Node: " + self.identifier + "\n")

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        parallel_results = my_pool.starmap(mixed_single_relu_forward, zip(abs_input.stars,
                                                                          itertools.repeat(self.neuron_relevance),
                                                                          itertools.repeat(self.iqr_thresholding),
                                                                          itertools.repeat(self.iqr_mult),
                                                                          itertools.repeat(self.refinement_percentage)))
        my_pool.close()

        abs_output = StarSet()
        self.overapprox_areas = []
        for result in parallel_results:
            star_areas = result[1]
            self.overapprox_areas.append(star_areas)
            star_set = result[0]
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    def __starset_forward(self, abs_input: StarSet) -> StarSet:
        """
        Forward function specialized for the concrete AbsElement StarSet.
        """

        abs_output = StarSet()
        for star in abs_input.stars:

            results = mixed_single_relu_forward(star, self.neuron_relevance, self.iqr_thresholding,
                                                self.iqr_mult, self.refinement_percentage)

            abs_output.stars = abs_output.stars.union(results[0])

        return abs_output


class AbsFullyConnectedNode(AbsLayerNode):
    """
    A class used for our internal representation of a Fully Connected Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : FullyConnectedNode
        LayerNode di riferimento per l'abstract transformer.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction.
    """

    def __init__(self, identifier: str, ref_node: nodes.FullyConnectedNode):
        super().__init__(identifier, ref_node)

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # Need to expand bias since they are memorized like one-dimensional vector in fullyconnectednodes.
        if self.ref_node.bias.shape != (self.ref_node.weight.shape[0], 1):
            bias = np.expand_dims(self.ref_node.bias, 1)
        else:
            bias = self.ref_node.bias
        parallel_results = my_pool.starmap(single_fc_forward, zip(abs_input.stars,
                                                                  itertools.repeat(self.ref_node.weight),
                                                                  itertools.repeat(bias)))
        my_pool.close()
        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsSigmoidNode(AbsLayerNode):
    """
    A class used for our internal representation of a Sigmoid transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : SigmoidNode
        LayerNode di riferimento per l'abstract transformer.

    refinement_level : Union[int, List[int]]

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction.
    """

    def __init__(self, identifier: str, ref_node: nodes.SigmoidNode, approx_levels: Union[int, List[int]] = None):
        super().__init__(identifier, ref_node)

        if approx_levels is None:
            approx_levels = [0 for i in range(ref_node.num_features)]
        elif isinstance(approx_levels, int):
            approx_levels = [approx_levels for i in range(ref_node.num_features)]

        self.approx_levels = approx_levels

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        parallel = True
        if parallel:
            abs_output = StarSet()
            my_pool = multiprocessing.Pool(1)
            parallel_results = my_pool.starmap(single_sigmoid_forward, zip(abs_input.stars,
                                                                           itertools.repeat(self.approx_levels)))
            my_pool.close()
            for star_set in parallel_results:
                abs_output.stars = abs_output.stars.union(star_set)
        else:
            abs_output = StarSet()
            for star in abs_input.stars:
                abs_output.stars = abs_output.stars.union(single_sigmoid_forward(star, self.approx_levels))

        return abs_output

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsNeuralNetwork(abc.ABC):
    """
    An abstract class used for our internal representation of a generic NeuralNetwork for Abstract Interpretation.
    It consists of a graph of AbsLayerNodes. The properties of the computational graph are specialized in the
    concrete classes. The method forward and backward calls the corresponding methods in the AbsLayerNodes following the
    correct order to compute the output AbsElement.

    Attributes
    ----------
    nodes : dict <str, LayerNode>
        Dictionary containing str keys and AbsLayerNodes values. It contains the nodes of the graph,
        the identifier of the node of interest is used as a key in the nodes dictionary.

    edges : dict <str, list <str>>
        Dictionary of identifiers of AbsLayerNodes, it contains for each nodes identified by the keys, the list of nodes
        connected to it.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction.

    """

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        pass

    @abc.abstractmethod
    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsSeqNetwork(AbsNeuralNetwork):
    """
    Concrete children of AbsNeuralNetwork representing a sequential AbsNeuralNetwork.
    It consists of a graph of LayerNodes. The computational graph of a SequentialNetwork must
    correspond to a standard list. The method forward and backward calls the corresponding methods
    in the AbsLayerNodes following the correct order to compute the output AbsElement.

    Attributes
    ----------
    identifier : str
        Identifier of the Sequential AbsNeuralNetwork.

    Methods
    -------
    add_node(LayerNode)
        Procedure to add a new AbsLayerNode to the sequential AbsNeuralNetwork.

    get_first_node()
        Procedure to extract the first AbsLayerNode of the sequential AbsNeuralNetwork.

    get_next_node(LayerNode)
        Procedure to get the next AbsLayerNode of the AbsNeuralNetwork given an input AbsLayerNode

    get_last_node()
        Procedure to extract the last AbsLayerNode of the sequential AbsNeuralNetwork.

    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction.

    """

    def __init__(self, identifier: str):

        super().__init__()
        self.identifier = identifier

    def add_node(self, node: AbsLayerNode):
        """
        Procedure to add a new AbsLayerNode. In sequential network the new node must be connected directly to the
        previous node forming a list.

        Parameters
        ----------
        node : AbsLayerNode
            New node to add to the Sequential network.

        """

        if len(self.nodes.keys()) == 0:
            self.nodes[node.identifier] = node
            self.edges[node.identifier] = []
        else:
            previous_node_key = self.get_last_node().identifier
            self.nodes[node.identifier] = node
            self.edges[previous_node_key].append(node.identifier)
            self.edges[node.identifier] = []

    def get_first_node(self) -> AbsLayerNode:
        """
        Procedure to get the first AbsLayerNode of the network.

        Return
        ---------
        AbsLayerNode
            The first node of the network.

        """
        assert self.nodes

        keys = [key for key in self.nodes.keys()]
        for key in self.nodes.keys():
            for sub_key in self.nodes.keys():
                if sub_key in self.edges[key]:
                    keys.remove(sub_key)

        return self.nodes[keys[0]]

    def get_next_node(self, node: AbsLayerNode) -> AbsLayerNode:
        """
        Procedure to get the next AbsLayerNode of the network given an input AbsLayerNode.

        Return
        ---------
        LayerNode
            The next node of the network.

        """

        assert self.nodes

        next_node = None
        if node is not None:
            current_key = node.identifier
            if len(self.edges[current_key]) != 0:
                next_key = self.edges[current_key][0]
                next_node = self.nodes[next_key]
        else:
            next_node = self.get_first_node()

        return next_node

    def get_last_node(self) -> AbsLayerNode:
        """
        Procedure to get the last AbsLayerNode of the network.

        Return
        ---------
        AbsLayerNode
            The last node of the network.

        """

        assert self.nodes

        current_node = self.get_first_node()
        while self.get_next_node(current_node) is not None:
            current_node = self.get_next_node(current_node)

        return current_node

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        current_node = self.get_first_node()
        while current_node is not None:
            abs_input = current_node.forward(abs_input)
            current_node = self.get_next_node(current_node)

        return abs_input

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass
