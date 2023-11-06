import torch
from torch import Tensor

def information_quantity(vector: Tensor) -> Tensor:
    """
    Calculate the information quantity of a vector, defined as the dot product of the vector with itself.
    This can be interpreted as a measure of the vector's magnitude squared.

    Args:
        vector (Tensor): A tensor representing the vector whose information quantity is to be calculated.

    Returns:
        Tensor: A single-element tensor containing the information quantity of the vector.
    """
    inf_quant = torch.dot(vector, vector)
    return inf_quant


def generalisation(vector_1: Tensor, vector_2: Tensor, op: int) -> Tensor:
    """
    Perform a generalisation operation on two vectors based on the specified function mode.

    Args:
        vector_1 (Tensor): The first input tensor.
        vector_2 (Tensor): The second input tensor.
        fun (int): An integer flag to determine the mode of operation.
            - If 1, perform a scalar product-based combination.
            - If 2, perform a minimum element-based combination.
            - If 3, take the minimum element of the two vectors.

    Returns:
        Tensor: The resulting tensor after applying the generalisation operation.

    Raises:
        ValueError: If `fun` is not in [1, 2, 3].
    """

    if  op == 1:
        scalar_product = torch.dot(vector_1, vector_2)
        squared_module_1 = torch.dot(vector_1, vector_1)
        squared_module_2 = torch.dot(vector_2, vector_2)
        generalized_vector = scalar_product * (vector_1 / squared_module_1 +  
                                               vector_2 / squared_module_2) * 0.5
    elif op == 2:
        module_1 = torch.norm(vector_1)
        module_2 = torch.norm(vector_2)
        vector_module = min(module_1.item(), module_2.item()) / max(module_1.item(), module_2.item())

        min_elems = torch.min(torch.abs(vector_1), torch.abs(vector_2))
        signs = torch.where((vector_1 == min_elems) | (vector_2 == min_elems), 1, -1)
        generalized_vector = signs * torch.sqrt(vector_module * torch.abs(vector_1 * vector_2))

    elif op == 3:
        min_elems = torch.min(torch.abs(vector_1), torch.abs(vector_2))
        signs = torch.where((vector_1 == min_elems) | (vector_2 == min_elems), 1, -1)
        generalized_vector = signs * min_elems
    
    else:
       raise ValueError('Unsupported operation: ', op)

    return generalized_vector


def specification(vector_1: Tensor, vector_2: Tensor, op: int) -> Tensor:
    """
    Perform a specification operation on two vectors based on the specified function mode.

    Args:
        vector_1 (Tensor): The first input tensor.
        vector_2 (Tensor): The second input tensor.
        fun (int): An integer flag to determine the mode of operation.
            - If 1, simply add the two vectors.
            - If 2, combine the vectors based on the maximum element.
            - If 3, take the maximum element of the two vectors.

    Returns:
        Tensor: The resulting tensor after applying the specification operation.

    Raises:
        ValueError: If `fun` is not in [1, 2, 3].
    """

    if  op == 1:
        specified_vector = vector_1 + vector_2

    elif op == 2:
        module_1 = torch.norm(vector_1)
        module_2 = torch.norm(vector_2)
        vector_module = min(module_1.item(), module_2.item()) / max(module_1.item(), module_2.item())

        max_elems = torch.max(torch.abs(vector_1), torch.abs(vector_2))
        signs = torch.where((vector_1 == max_elems) | (vector_2 == max_elems), 1, -1)
        specified_vector = signs * torch.sqrt(vector_1**2 + vector_2**2 - 
                                              vector_module * torch.abs(vector_1 * vector_2))
    elif op == 3:
        max_elems = torch.max(torch.abs(vector_1), torch.abs(vector_2))
        signs = torch.where((vector_1 == max_elems) | (vector_2 == max_elems), 1, -1)
        specified_vector = signs * max_elems
    
    else:
        raise ValueError('Unsupported operation: ', op)
    
    return specified_vector

