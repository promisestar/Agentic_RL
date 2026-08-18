# tests/test_embedding_model.py
import logging
import os

# 让 HF Hub 的日志可见，至少能知道在做什么
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

# 防止每次偶然尝试再发请求
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

try:
    m = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True, # ← 找不到直接报错，而不是默默发请求
    )
    print("OK", m.device, m.get_sentence_embedding_dimension())
except Exception as e:
    print("FAIL", type(e).__name__, e)