import torch 
from llama.model import LLaMA

model = LLaMA.from_pretrained("/Users/alf/llama/llama-2-13b-chat")
model.eval()

text = "Hello World how are you today??"
inputs = model.tokenizer(text,return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.embeddings

print(embeddings.shape)