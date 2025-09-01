from new_langchaing_practice.embeddings import bge_hf_model

dim  = len(bge_hf_model.embed_query("Hello"))
print(dim)
# print(len(res))