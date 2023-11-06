
import torch

from torch import Tensor 
from typing import Union, Tuple


def f_fun(alpha: Union[int, float], mu: Union[int, float], vector_1:Tensor, vector_2:Tensor) -> Tensor:
    """
    Apply a composite function to two vectors using specified parameters.

    Args:
        alpha (float or Tensor): The weight for the sum of squared norms component of the function.
        mu (float or Tensor): The weight for the dot product component of the function.
        vector_1 (Tensor): The first vector in the operation.
        vector_2 (Tensor): The second vector in the operation.

    Returns:
        Tensor: The resulting vector after applying the composite function to `vector_1` and `vector_2`.
    """

    # Compute the normalized sum of the vectors
    normalized_sum = (vector_1 + vector_2) / torch.norm(vector_1 + vector_2)

    # Calculate the weighted sum of the squared norms
    weighted_norms = alpha * (torch.dot(vector_1, vector_1) + torch.dot(vector_2, vector_2))

    # Calculate the weighted dot product
    weighted_dot_product = mu * torch.dot(vector_1, vector_2)

    # Compute the rightmost operation with separated weighted components
    right_op = torch.sqrt(weighted_norms - weighted_dot_product)

    return normalized_sum * right_op


def parameters(vector_1:Tensor, vector_2:Tensor, comp_fun:str) -> Tuple[Union[int, float], Union[int, float]]: 
    """
    Calculate parameters based on composite function identifier.

    Args:
        vector_1 (Tensor): The first vector in the operation.
        vector_2 (Tensor): The second vector in the operation.
        comp_fun (str): A string identifier for the composite function 
                        which determines the operation to perform. Accepted
                        values are "sum", "icds_avg", "ind", "jnt", and "inf".

    Returns:
        A tuple of two elements (alpha, mu) where:
    """

    op_dict = {
        "sum":                  (1, -2),
        "icds_avg":             (0.25, -0.5),
        "ind":                  (1, 0),
        "jnt":                  (1, 1),
        "inf": lambda v1, v2:   (1, (min(torch.norm(v1), torch.norm(v2)) /
                                     max(torch.norm(v1), torch.norm(v2))).item())
    }

    params = op_dict[comp_fun]
    # inf case
    if callable(params):  params = params(vector_1, vector_2)

    return params


def compose(vector_list:Union[Tensor, list], comp_fun:str) -> list:
    """
    Compose a list of vectors into a single vector based on:
        - Sum or average of the tensor
        - Sequential pairwise operation of the specified function.

    Args:
        vector_list (Tensor): A batch of vectors to be composed.
        comp_fun (str): The name of the composition function to apply. Supported functions
                        are 'sum', 'avg', and any key present in the `parameters` function's
                        operation dictionary.

    Returns:
        Tensor: The resulting composed vector after applying the specified composite function.

    Raises:
        KeyError: If the `comp_fun` is not recognized (not in the `parameters` function's
                  operation dictionary).

    Examples:
        >>> vector_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> compose(vector_list, 'sum')
        tensor([5, 7, 9])
        >>> compose(vector_list, 'avg')
        tensor([2.5, 3.5, 4.5])
    """
    if isinstance(vector_list, list): vector_list = torch.tensor(vector_list)

    if not len(vector_list):  return torch.tensor([])
    if comp_fun == 'sum':     return torch.sum(vector_list, dim = 0)
    if comp_fun == 'avg':     return (torch.sum(vector_list, dim = 0) / len(vector_list))
    
    # ICDS cases
    vector_1 = vector_list[0]
    for vector_2 in vector_list[1:]:
        alpha, mu = parameters(vector_1, vector_2, comp_fun)
        vector_1 = f_fun(alpha, mu, vector_1, vector_2)
    
    output = vector_1
    return output
                

### TODO: Batched composition with torch GPU vectorization
### Work in progress, do not use.
def generalized_composition_function(alpha: float, beta: float, vector_1: Tensor, vector_2: Tensor) -> Tensor:
    
    # Calculate the sum of the input vectors
    vector_sum = vector_1 + vector_2

    # Calculate the norm of the sum of the input vectors
    norm_vector_sum = torch.linalg.norm(vector_sum, dim=-1)

    # Calculate the first part of the generalized composition function
    alpha_part = alpha * (torch.linalg.norm(vector_1, dim=-1)**2 + torch.linalg.norm(vector_2, dim = -1)**2)

    # Calculate the second part of the generalized composition function
    beta_part = beta * torch.bmm(vector_2.unsqueeze(1), vector_1.unsqueeze(2)).squeeze()

    # Calculate the square root of the absolute difference between the two parts
    square_root = torch.sqrt(torch.abs(alpha_part - beta_part))

    # Calculate the composite vector using the calculated values
    composite_vector = vector_sum / norm_vector_sum.unsqueeze(-1) * square_root.unsqueeze(-1)

    return composite_vector

def f(operation: str, vector_1: Tensor, vector_2: Tensor) -> Tensor:
    # Dictionary with the possible operations and their parameters
    op_dict = {
        "sum": (1, -2),
        "avg": (0.25, -0.5),
        "ind": (1, 0),
        "jnt": (1, 1),
        "inf": (1, calc_beta_finf(vector_1, vector_2)),
    }

    return generalized_composition_function(*op_dict[operation[:3]], vector_1, vector_2)


def calc_beta_finf(vector_1: Tensor, vector_2: Tensor) -> float:
    
    # Calculate the minimum and maximum norm of the input vectors
    v1 = torch.linalg.norm(vector_1, dim=-1)
    v2 = torch.linalg.norm(vector_2, dim=-1)
    condition = v1 > v2
    minimum = torch.where(condition, v2, v1)
    maximum = torch.where(condition, v1, v2)

    # Calculate the beta value
    beta_finf = minimum / maximum

    return beta_finf


def get_composite_vector(operation: str, embeddings_list: Tensor, attention_mask: Tensor, r2l: bool = False):
    # Get the index of the last token that is not padding
    result_index = (torch.sum(attention_mask, dim=1) - 1).long()
    #embeddings_list = embeddings_list.clone()
    results = embeddings_list.clone()
    # Reverse the subtensors containing token representations if r2l is True
    if r2l:
        for i in range(len(embeddings_list)):
            results[i, :result_index[i]+1] = torch.flip(embeddings_list[i, :result_index[i]+1], [0]).clone()
    # Cummulative batch vectorization of composition function
    for i in range(1,embeddings_list.shape[1]):        
        results[:,i,:] = f(operation, results[:,i-1,:].clone(), results[:,i,:].clone())

    # Gather cummulative result of the last token that is not padding 
    output = results[torch.arange(results.shape[0]), result_index]
    return output