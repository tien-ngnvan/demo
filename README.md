# demo

## How to use
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