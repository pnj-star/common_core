import common_core.providers.cache as cache_module

from common_core.config import CacheConfig, LLMConfig
from common_core.providers import (
    OpenAICompatibleLLM,
    RedisCache,
    build_filter_expr,
    response_ttl_for,
    rrf_fuse,
)


def test_rrf_fuse_ranks_docs_hit_by_both_routes_first() -> None:
    dense = [
        {"id": "a", "content": "a", "score": 0.9},
        {"id": "b", "content": "b", "score": 0.8},
    ]
    sparse = [
        {"id": "b", "content": "b", "bm25_score": 2.0},
        {"id": "c", "content": "c", "bm25_score": 1.0},
    ]
    fused = rrf_fuse(dense, sparse)
    assert [doc["id"] for doc in fused] == ["b", "a", "c"]
    assert "fusion_score" in fused[0]
    assert "dense_rank" in fused[0]
    assert "sparse_rank" in fused[0]


def test_build_filter_expr_handles_scalar_and_list_filters() -> None:
    expr = build_filter_expr(
        {
            "category": "knowledge",
            "product_id": ["p1", "p2"],
            "tag": ["x", "y"],
        }
    )
    assert 'category == "knowledge"' in expr
    assert '"p1", "p2"' in expr
    assert expr.startswith("category == ")


def test_redis_cache_key_is_tenant_and_kb_scoped() -> None:
    cache = RedisCache(CacheConfig(key_prefix="shop"))
    key_a = cache.key("resp", "hello world", tenant_id="t1", kb_id="kb1")
    key_b = cache.key("resp", "hello world", tenant_id="t1", kb_id="kb2")
    key_c = cache.key("resp", "hello world")
    assert key_a.startswith("shop:resp:t1:kb1:")
    assert key_a != key_b
    assert key_a != key_c


def test_response_ttl_shortens_truncated_replies() -> None:
    assert response_ttl_for("ok") == 300
    assert response_ttl_for("a reasonably complete answer") == 1800


def test_llm_client_holds_config_without_importing_openai() -> None:
    llm = OpenAICompatibleLLM(
        LLMConfig(base_url="http://llm.local/v1", model="generic-model")
    )
    assert llm.config.base_url == "http://llm.local/v1"
    assert llm.config.model == "generic-model"


def test_redis_cache_shared_mode_reuses_client(monkeypatch) -> None:
    created: list[object] = []

    def fake_factory(config):
        created.append(config)
        return object()

    monkeypatch.setattr(cache_module, "_shared_clients", {})
    monkeypatch.setattr(cache_module, "_create_redis_client", fake_factory)
    first = RedisCache(CacheConfig(host="cache", port=6379), shared=True)
    second = RedisCache(CacheConfig(host="cache", port=6379), shared=True)
    assert first.client() is second.client()
    assert len(created) == 1

    standalone = RedisCache(CacheConfig(host="cache", port=6379))
    assert standalone.client() is not first.client()
    assert len(created) == 2


def test_milvus_to_docs_reads_milvus_client_dict_hits() -> None:
    from common_core.providers.vector import MilvusVectorStore

    store = MilvusVectorStore()
    docs = store._to_docs(
        [[{"id": "1", "distance": 0.9, "content": "hello", "parent_id": "p1"}]],
        ["id", "content", "parent_id"],
        "score",
    )
    assert docs == [
        {"id": "1", "score": 0.9, "content": "hello", "parent_id": "p1"}
    ]


class _FakeMilvusClient:
    """模拟 MilvusClient 的 search/has_collection/load_collection 接口。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def has_collection(self, collection_name: str) -> bool:
        return True

    def load_collection(self, collection_name: str) -> None:
        self.loaded = collection_name

    def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
        self.calls.append(kwargs)
        return [[{"id": "1", "distance": 0.9, "content": "x"}]]


def test_milvus_dense_search_uses_client_api() -> None:
    from common_core.providers.vector import MilvusVectorStore

    store = MilvusVectorStore()
    fake = _FakeMilvusClient()
    store._client = fake
    store._loaded_collections = {"kb"}

    docs = store.search_dense("kb", [0.1, 0.2], output_fields=["id", "content"])

    assert docs[0]["id"] == "1"
    assert docs[0]["content"] == "x"
    call = fake.calls[0]
    assert call["collection_name"] == "kb"
    assert call["anns_field"] == "embedding"
    assert call["search_params"] == {
        "metric_type": "COSINE",
        "params": {"nprobe": 16},
    }
    assert call["data"] == [[0.1, 0.2]]


def test_milvus_bm25_search_uses_client_api() -> None:
    from common_core.providers.vector import MilvusVectorStore

    store = MilvusVectorStore()
    fake = _FakeMilvusClient()
    store._client = fake
    store._loaded_collections = {"kb"}

    docs = store.search_bm25("kb", "how to apply", output_fields=["id"])

    assert docs[0]["id"] == "1"
    call = fake.calls[0]
    assert call["anns_field"] == "sparse"
    assert call["search_params"] == {"metric_type": "BM25"}
    assert call["data"] == ["how to apply"]
