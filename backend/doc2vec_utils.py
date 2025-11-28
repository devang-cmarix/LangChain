# backend/doc2vec_utils.py
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict, Any
import os
import multiprocessing
import re

def simple_tokenize(text: str):
    # light tokenization: lower, remove multiple spaces, split on non-alphanum
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"\w+", text)
    return tokens

def prepare_tagged_documents(docs: List[Dict[str, Any]]):
    """
    docs: list of dicts like {"id": some_unique_id, "text": "..."}
    Return: list of TaggedDocument
    """
    tagged = []
    for d in docs:
        tokens = simple_tokenize(d["text"])
        tagged.append(TaggedDocument(words=tokens, tags=[str(d["id"])]))
    return tagged

def train_doc2vec(tagged_docs, *, vector_size=128, window=8, min_count=2, epochs=40, dm=1, workers=None):
    """
    Train a Doc2Vec model on TaggedDocument list.
      - dm=1 -> distributed memory (PV-DM)
      - dm=0 -> dbow (PV-DBOW)
    """
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        dm=dm,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def save_model(model: Doc2Vec, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_model(path: str) -> Doc2Vec:
    return Doc2Vec.load(path)

def infer_vector(model: Doc2Vec, text: str, steps=20):
    tokens = simple_tokenize(text)
    return model.infer_vector(tokens, steps=steps)
