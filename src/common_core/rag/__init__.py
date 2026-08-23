"""可复用的 RAG 回答侧公共机制（供 skill 之外的 agent / 多 agent 框架复用）。

本子包存放与"把检索到的文档变成最终回答"相关的通用库函数，不含业务逻辑：
上下文组装（assembly）、回答生成（generation）与输出质量护栏（guard）。
检索本身属于 skill 的职责，这里只提供回答侧的编排原料。
"""

from .assembly import (
    DEFAULT_MAX_CONTEXT_CHARS,
    build_context_text,
    clean_markdown,
    dedupe_docs,
    extract_images,
)
from .generation import (
    DEFAULT_FALLBACK_PROMPT_TEMPLATE,
    DEFAULT_FALLBACK_RESPONSE,
    DEFAULT_PROMPT_TEMPLATE,
    GenerationConfig,
    build_messages,
    generate_answer,
    stream_answer,
)
from .guard import (
    DEFAULT_ABSOLUTE_WORDS,
    DEFAULT_REVIEW_PROMPT,
    GuardConfig,
    GuardResult,
    absolute_language_issues,
    check_compound_numbers,
    evaluate_guard,
    extract_all_numbers,
    extract_risky_numbers,
    guard_generation,
)

__all__ = [
    "DEFAULT_ABSOLUTE_WORDS",
    "DEFAULT_FALLBACK_PROMPT_TEMPLATE",
    "DEFAULT_FALLBACK_RESPONSE",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_REVIEW_PROMPT",
    "GenerationConfig",
    "GuardConfig",
    "GuardResult",
    "absolute_language_issues",
    "build_context_text",
    "build_messages",
    "check_compound_numbers",
    "clean_markdown",
    "dedupe_docs",
    "evaluate_guard",
    "extract_all_numbers",
    "extract_images",
    "extract_risky_numbers",
    "generate_answer",
    "guard_generation",
    "stream_answer",
]
