import torch 
import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 分词示例
text = "Hello, how are you?Hello, how are you?" # （batch_size, 序列长度, 隐藏层维度）
encoded = tokenizer(text, padding="max_length",max_length=128, return_tensors="pt")

# 前向传播获取编码向量
with torch.no_grad():
    outputs = model(**encoded)

print(outputs.last_hidden_state.shape)
# 输出维度示例：[1, 8, 768] （batch_size, 序列长度, 隐藏层维度）
