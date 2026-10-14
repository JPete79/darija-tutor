"""List Hugging Face repos that look like GGUF builds of Atlas-Chat / GemMaroc."""
from huggingface_hub import HfApi

api = HfApi()
for query in ("Atlas-Chat", "GemMaroc"):
    for m in api.list_models(search=query, limit=50):
        if "gguf" in m.id.lower() or "gguf" in (m.tags or []):
            print(m.id)
