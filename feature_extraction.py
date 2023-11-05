import modules.composition

from transformers import AutoModel, AutoTokenizer, pipeline

if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline('composition', model=model, tokenizer= tokenizer)

    print(pipe('hi there', comp_fun='inf'))
    