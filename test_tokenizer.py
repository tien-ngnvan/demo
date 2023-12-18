import os
from hf_tokenizer import PhobertTokenizer
from pyvi import ViTokenizer

model_name_or_path = 'vietnamese-bi-encoder'
tokenizer = PhobertTokenizer.from_pretrained(
    model_name_or_path,
    vocab_file=os.path.join(model_name_or_path,'vocab.txt'),
    merges_file=os.path.join(model_name_or_path,'bpe.codes'))

pyvi_tokened = ViTokenizer.tokenize('Hello, đại học Công Nghệ Thông Tin')
print("Tokenize: ", tokenizer(pyvi_tokened))