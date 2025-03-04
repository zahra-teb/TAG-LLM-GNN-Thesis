from transformers import AutoModel, AutoTokenizer
import torch

model_name = "bert-base-uncased" # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

def encode_texts(texts):
    """
    Convert raw texts to embeddings.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def generate_embeddings(texts, explanations):
    """
    Generate node embeddings from both original texts and explanations.
    """
    h_orig = encode_texts(texts)
    h_expl = encode_texts(explanations)
    return h_orig, h_expl
