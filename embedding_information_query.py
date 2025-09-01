import ast

import numpy as np
import pandas as pd

from new_langchaing_practice.embeddings import bge_hf_model

def text_to_embedding(text):
    resp = bge_hf_model.embed_documents(
        [text]
    )
    return resp[0]

def embedding_to_file(source_file, output_file):
    """
    Read original review test. Use embedding model to convert them to vectors and same them to new file
    """
    df = pd.read_csv(source_file, index_col = 0)
    df = df[["Summary", "Text"]]
    print(df.head())

    df = df.dropna()

    df["text_content"] = "Summary: " + df["Summary"].str.strip() + "; Text: " + df["Text"].str.strip()
    print(df.head())

    # Embed
    df["embedding"] = df["text_content"].apply(lambda x: text_to_embedding(x))
    df.to_csv(output_file)

def cosine_distance(a,b):
    """
    compute cosine distance
    """
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

def search_text(input, embedding_file, top_n = 3):
    """
    retrieval information based on user's question. Return n most relevant results.
    """
    df_data = pd.read_csv(embedding_file)
    # Convert string back to vectors
    df_data["embedding_vector"] = df_data["embedding"].apply(ast.literal_eval)

    input_vector = text_to_embedding(input)

    df_data['Similarity'] = df_data['embedding_vector'].apply(lambda x : cosine_distance(x, input_vector))

    res = (
        df_data.sort_values("Similarity", ascending = False)
        .head(top_n)
        .text_content.str.replace("Summary: ", "").str.replace("; Text: ", "")
    )
    for r in res:
        print(r)
        print("-"*30)


if __name__=="__main__":
    # embedding_to_file("./data/fine_food_reviews_1k.csv", "./data/output.csv")
    search_text("delicious beans", "./data/output.csv", top_n=3)





