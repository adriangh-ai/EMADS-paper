import torch

from modules import composition, comp_func
from transformers import AutoModel, AutoTokenizer, pipeline

if __name__ == "__main__":
    # Model instantiation
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Pipeline instantiation
    pipe = pipeline('composition', model=model, tokenizer= tokenizer)

    # Composition of the model representation of a sentence 
    output = pipe('this is a sentence', comp_fun='sum')
    print(output)

    # Composition of a sentence by its constinuent words
    output = pipe('this is a sentence'.split(), comp_fun='sum')
    output = comp_func.compose(output, comp_fun = 'avg')
    print(output)

