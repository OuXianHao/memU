from __future__ import annotations

import importlib.util
from pathlib import Path


def test_qwen3_memu_demo_imports_without_running_main():
    script = Path(__file__).parent.parent / "examples" / "qwen3_memu_demo.py"
    spec = importlib.util.spec_from_file_location("qwen3_memu_demo", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.QWEN_BASE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert module.QWEN_MODEL == "Qwen3-30B-A3B-Instruct-2507"
    assert (
        module.format_memory_context({"segments": [{"text": "prefers reproducible experiments"}]})
        == "- prefers reproducible experiments"
    )
