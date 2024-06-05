# 大模型推理框架

## 指标
latency
thoughtput(req/s): 每秒请求数=请求总数 / 总时间

## vLLM
适用于大批量Prompt输入，并对推理速度要求高的场景，支持较高的吞吐量。
底层使用 ray 做调度。

```bash
python -m vllm.entrypoints.api_server --model /xxx/ --tensor-parallel-size 2 --port 8000
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
models = client.models.list()
model = models.data[0].id
chat_response = client.chat.completions.create(
    messages=[{"role": "user", "content": "你好"}], model=model, stream=False)
```

## Text generation inference
Huggingface 官方推理框架，对自己的模型支持较好。
主要包含 launcher（启动器）、router（服务的中间件）、serve（gRPC 服务）三个部分。

```bash
text-generation-launcher --model-id /xxx/ --port 8000
```

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://127.0.0.1:8000")
client.text_generation(prompt="你好", stream=False)
```

## LMDeploy
请求数及推理速度都优于 vLLM（官方介绍）

```bash
lmdeploy serve api_server /xxx/ --server-port 8000 --tp 4 --model-name llama
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="test")

completion = client.chat.completions.create(
    model="llama", messages=[{"role": "user", "content": "你好"}], stream=False)
```

## Ollama

## DeepSpeed-FastGen


