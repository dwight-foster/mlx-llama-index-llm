# MLX LLama-Index LLM

MLX LLama-Index LLM is a llama-index LLM integration for the [MLX](https://github.com/ml-explore/mlx) machine learning framework. It can be used the same as other [llama-index](https://github.com/run-llama/llama_index) llms to work seamlessy with tools such as RAG.

## Features

- Seamless Integration: Easily integrates with the MLX machine learning framework.
- Compatibility: Works with existing llama-index tools and libraries.
- High Performance: Optimized for efficiency and speed.
- Extensible: Easily extendable to add new features or modify existing ones.

## Installation
```
git clone https://github.com/yourusername/mlx-llama-index-llm.git
cd mlx-llama-index-llm
pip install -r requirements.txt
```

## Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from mlx_lm import load, generate, convert
from base import MLXLLM
documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
model, tokenizer = load("mlx_model_quantized")
llm = MLXLLM(model=model, tokenizer=tokenizer)
# ollama
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("What is minimax?")

print(response)
```

## Acknowledgments
- [MLX](https://github.com/ml-explore/mlx) - The MLX library for machine learning on Apple devices
- [Llama-Index](https://github.com/run-llama/llama_index) - The llama-index library that I based my llm off of

  
