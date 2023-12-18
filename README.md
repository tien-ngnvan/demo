# demo
## Installed
`pip install -r requirements.txt`

## 1. How to use with Wordpiece (bert and variants)
```python
import os
from hf_tokenizer import BertTokenizer


model_name_or_path = 'openvino_mBert'
tokenizer = BertTokenizer.from_pretrained(
    model_name_or_path, os.path.join(model_name_or_path,'vocab.txt')
)

print("Tokenize: ", tokenizer.tokenize('hello tháng 12'))
print("Encode: ", tokenizer.encode('hello tháng 12'))
print("Decode: ", tokenizer.decode(tokenizer.encode('hello tháng 12')))

# Tokenize:  ['hell', '##o', 'tháng', '12']
# Encode:  [101, 61694, 10133, 11642, 10186, 102]
# Decode:  [CLS] hello tháng 12 [SEP]
```

## 2. How to use with PhoBert (bpe and variants)
First, clone `vietnamese-bi-encoder` from huggingface hub

`git clone https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder`

and get `vocab.txt` support from phobert-base-2
```
%cd vietnamese-bi-encoder
wget https://huggingface.co/vinai/phobert-base-v2/resolve/main/tokenizer.json
```
Because Phobert need segment words before tokenize, so we need install `pyvi` to tokenize segment. 

```python
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

# Tokenize:  {'input_ids': [0, 29048, 4, 956, 3793, 27870, 12422, 2858, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```