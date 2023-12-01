import torch
import logging

from typing import List, Dict
from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction


logger = logging.getLogger()


class OVModel:
    def __init__(self, onnx_dir) -> None:
        self.model = OVModelForFeatureExtraction.from_pretrained(onnx_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

        logger.info(self.model)
        
    def embed_query(self, sentence:str=None, q_max_length:int = 256) -> Dict[str, float]:
        tokenized = self.get_tokenized(sentences=sentence, max_length=q_max_length)
        with torch.no_grad():
            model_output = self.model(**tokenized)
            hidden_state = self.get_cls(model_output)
            embeddings = hidden_state.cpu().detach().numpy()

        if embeddings.shape[0] == 1:
            return list(embeddings.reshape(-1))
        else:
            return [list(embd.reshape(-1)) for embd in embeddings]

    def embed_documents(self, sentences:List[str], p_max_length:int = 384):
        tokenized = self.get_tokenized(sentences=sentences, max_length=p_max_length)
        with torch.no_grad():
            model_outputs = self.model(**tokenized)
            hidden_state = self.get_cls(model_outputs)
            embeddings = hidden_state.cpu().detach().numpy()

        if embeddings.shape[0] == 1:
            return list(embeddings.reshape(-1))
        else:
            return [list(embd.reshape(-1)) for embd in embeddings]
        

    def get_tokenized(self, sentences, max_length):
        tokenized = self.tokenizer(sentences, max_length=max_length,
                                   padding='max_length', truncation=True,
                                   return_tensors='pt')

        return tokenized

    def get_cls(self, x):
        cls = x.last_hidden_state[:, 0]

        return cls