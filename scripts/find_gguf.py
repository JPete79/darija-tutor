"""List GGUF builds of Darija LLMs on Hugging Face, or the files inside one repo.

    python scripts/find_gguf.py                       # candidate repos, most downloaded first
    python scripts/find_gguf.py <repo_id>             # list .gguf files and sizes in a repo
"""
import sys

from huggingface_hub import HfApi

QUERIES = ("Atlas-Chat", "GemMaroc", "darija gguf", "moroccan gguf")

api = HfApi()


def list_candidates() -> None:
    seen = {}
    for query in QUERIES:
        for m in api.list_models(search=query, limit=100, sort="downloads"):
            if "gguf" in m.id.lower() or "gguf" in (m.tags or []):
                seen[m.id] = m.downloads or 0
    for repo, downloads in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"{downloads:>7}  {repo}")


def list_files(repo: str) -> None:
    info = api.model_info(repo, files_metadata=True)
    for f in sorted(info.siblings, key=lambda s: s.rfilename):
        if f.rfilename.endswith(".gguf"):
            print(f"{(f.size or 0) / 1e9:>6.2f} GB  {f.rfilename}")


if __name__ == "__main__":
    list_files(sys.argv[1]) if len(sys.argv) > 1 else list_candidates()
