"""Use memU retrieval as memory context for Qwen3 generation.

Prerequisites:
    export QWEN_API_KEY=your_dashscope_key
    export MEMU_LOCAL_EMBED_MODEL=/path/to/bge-base-zh-v1.5

memU only stores and retrieves memory in this demo. The Qwen call is kept in
this external script so generation remains outside memU core.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import OpenAI

from memu.app import MemoryService

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "Qwen3-30B-A3B-Instruct-2507"


def format_memory_context(retrieval: dict[str, Any]) -> str:
    """Convert memU retrieval results into a compact prompt section."""
    lines: list[str] = []
    for segment in retrieval.get("segments", []):
        lines.append(f"- {segment['text']}")
    for resource in retrieval.get("resources", []):
        caption = resource.get("caption") or "source"
        lines.append(f"- {caption}: {resource['url']}")
    return "\n".join(lines) or "No relevant memory found."


async def main() -> None:
    embedding_model = os.environ.get("MEMU_LOCAL_EMBED_MODEL", "/path/to/bge-base-zh-v1.5")
    qwen_api_key = os.environ["QWEN_API_KEY"]

    memu = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
        embedding_profiles={"embedding": {"provider": "local", "model": embedding_model, "batch_size": 32}},
    )

    await memu.commit_results(
        recall_files=[
            {
                "name": "user-profile",
                "track": "memory",
                "description": "stable user preferences",
                "content": "# User profile\n用户喜欢可复现实验,并偏好中文向量模型。\n用户正在评估长期记忆智能体。",
            }
        ],
        resource=[{"path": "/experiments/locomo-notes.md", "description": "LoCoMo benchmark experiment notes"}],
    )

    query = "How should I configure the long-term memory experiment?"
    retrieval = await memu.progressive_retrieve(query)
    memory_context = format_memory_context(retrieval)

    prompt = f"""Relevant memory:
{memory_context}

User question:
{query}
"""

    qwen_client = OpenAI(api_key=qwen_api_key, base_url=QWEN_BASE_URL)
    response = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
