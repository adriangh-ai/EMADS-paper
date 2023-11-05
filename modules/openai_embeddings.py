import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_embed(text):
    output = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text,
                    encoding_format="float"
                )
    
    output = [embedding_data['embedding'] for embedding_data in output['data']]
    return output