# Embedding Meaning Algebra into Distributional Semantics

This repository contains the implementation for the work Embedding Meaning Algebra into
Distributional Semantics.

## Abstract
The field of distributional semantics has seen significant progress in recent years due
to advancements in natural language processing techniques, particularly through the
development of Neural Language Models like GPT and BERT. However, there are still
challenges to overcome in terms of semantic representation, particularly in the lack of
coherence and consistency in existing representation systems.
This work introduces a framework defining the relationship between a probabilistic
space, a set of meanings, and a vector space of static embedding representations; and
establishes formal properties based on definitions that would be desirable for any distri-
butional representation system to comply with in order to establish a common ground
between distributional semantics and other approaches. This work also introduces an
evaluation benchmark, defined on the basis of the formal properties introduced, which
will allow to measure the quality of a representation system.

## Usage

### Setting up the Environment

To use the code provided in this repository, you will need to have Python installed on your system along with the required libraries. Here are the steps to set up your environment:

1. Clone the repository to your local machine.
2. Install the required packages (recommended inside a venv):

    ```bash
    pip install -r requirements
    ```

### Running the Code

The main functionality of this repository is encapsulated in the `pipeline` for composing model representations of sentences using pre-trained models like BERT. Below are the steps to execute the core functionality:

1. Import the necessary classes and functions:

    ```python
    from modules import composition, comp_func # Important: loads the methods in HF Pipeline
    from transformers import AutoModel, AutoTokenizer, pipeline
    ```

2. Instantiate the model and tokenizer with a pre-trained BERT model:

    ```python
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

3. Create a pipeline for composition tasks:

    ```python
    pipe = pipeline('composition', model=model, tokenizer=tokenizer)
    ```

4. Compose the model representation of a sentence by summing up the embeddings:

    ```python
    output = pipe('this is a sentence', comp_fun='sum')
    ```

5. Alternatively, compose the representation of a sentence by its constituent words and then average the result:

    ```python
    output = pipe('this is a sentence'.split(), comp_fun='sum')
    output = comp_func.compose(output, comp_fun='avg')
    ```

The `comp_fun` argument determines the composition function used to combine word embeddings. The available functions are 'sum' and 'avg', which correspond to summing or averaging the embeddings, respectively.

Available funcions are: 
   - 'sum'         : Vector sum
   - 'avg'         : Global average
   - 'icds_avg'    : ICDS Average
   - 'ind'         : F independent
   - 'inf'         : F information
   - 'jnt'         : F joint

Please note that the actual `composition` functionality for the pipeline and the `comp_func.compose` method should be defined in your codebase as they are not standard functions in the transformers library.

### Output

The output will be a list containing the embeddings: list[list[float]]
