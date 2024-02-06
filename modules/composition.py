import torch
import modules.comp_func as cf

from transformers import Pipeline


class CompositionPipeline(Pipeline):
    """
    TODO: Pipline docstring
    """
    current_special_vector_mask = None

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        """
        Sanitize and prepare the parameters for tokenization and postprocessing.

        Args:
            truncation (bool, optional): Whether to truncate sequences to model's maximum length.
            tokenize_kwargs (dict, optional): Keyword arguments to pass to the tokenizer.
            return_tensors (bool, optional): Whether to return tensors instead of list of features.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - preprocess_params (dict): Parameters to pass to the tokenizer.
                - {} (dict): An empty dictionary reserved for future use.
                - postprocess_params (dict): Parameters to use during postprocessing, such as 'comp_fun'.

        Raises:
            ValueError: If `truncation` is specified both in `tokenize_kwargs` and as a parameter.
            ValueError: If 'batch_size' is specified, as it is not supported at the moment
        """

        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            
            tokenize_kwargs["truncation"] = truncation

        if 'bacth_size' in kwargs: raise ValueError('Batched inference is not currently supported.')
        
        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors
        if 'comp_fun' in kwargs:
            postprocess_params['comp_fun'] = kwargs['comp_fun']

        return preprocess_params, {}, postprocess_params


    def preprocess(self, inputs, **tokenize_kwargs):
        """
        Preprocess the input data for the model.

        Args:
            inputs (str or List[str]): One or several texts (or one list of texts) to tokenize.
            **tokenize_kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            dict: The tokenized and preprocessed inputs suitable for the model.
        """
        # Tokenize, adding EOS token for Causal Models
        # TODO Revise for corner cases (Encoders with EOS != PAD)
        return_tensors = self.framework
        model_inputs = self.tokenizer(
            inputs, 
            add_eos_token=True, 
            return_tensors=return_tensors, 
            **tokenize_kwargs
            )
        
        # Store current vector special token mask
        input_ids = model_inputs['input_ids'].squeeze()
        special_tokens_tensor = torch.tensor(self.tokenizer.all_special_ids)
        self.current_special_vector_mask = ~input_ids.unsqueeze(1).eq(special_tokens_tensor).any(1)
        
        return model_inputs


    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs


    def postprocess(self, model_outputs, comp_fun='sum', return_tensors=False):   
        """
        Postprocess the raw model outputs to remove special tokens and apply composition function.

        Args:
            model_outputs (Tensor): The raw outputs from the model forward pass.
            comp_fun (str, optional): The composition function to apply. Defaults to 'sum'.
            return_tensors (bool, optional): Whether to return tensors instead of list of features. 
                                             Defaults to False.

        Returns:
            list or Tensor: The postprocessed model outputs, with special tokens removed 
                            and composition function applied.
        """ 
        if return_tensors:        return model_outputs[0]
        

        # Remove special token vectors and compose
        output = model_outputs[0][0][self.current_special_vector_mask]
        if comp_fun in {'cls', 'eos'}: output = model_outputs.last_hidden_state[0]
        
        output = cf.compose(output, comp_fun)
        
        return output
    

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)