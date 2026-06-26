import sys
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from chunker import Chunk
from rag_chain import RAGChain
from vector_store import SimpleVectorStore


def make_chain_with_store(store):
    chain = object.__new__(RAGChain)
    chain.store = store
    return chain


def test_load_index_returns_true_for_existing_non_empty_index(tmp_path):
    index_file = tmp_path / "simple_index.json"
    store = SimpleVectorStore()
    store.add(
        [Chunk(content="Transformer 使用自注意力机制", metadata={"source": "t.md"})],
        [[0.1, 0.2, 0.3]],
    )
    store.save(str(index_file))

    chain = make_chain_with_store(SimpleVectorStore())

    assert chain.load_index(index_file) is True
    assert len(chain.store) == 1
    assert chain.store.chunks[0].content == "Transformer 使用自注意力机制"


def test_load_index_returns_false_for_missing_index(tmp_path):
    chain = make_chain_with_store(SimpleVectorStore())

    assert chain.load_index(tmp_path / "missing.json") is False
    assert len(chain.store) == 0


def test_load_index_returns_false_for_empty_index(tmp_path):
    index_file = tmp_path / "empty_index.json"
    SimpleVectorStore().save(str(index_file))
    chain = make_chain_with_store(SimpleVectorStore())

    assert chain.load_index(index_file) is False
    assert len(chain.store) == 0


def test_save_index_writes_store_to_file(tmp_path):
    index_file = tmp_path / "saved_index.json"
    store = SimpleVectorStore()
    store.add(
        [Chunk(content="RAG 会把检索结果放进 Prompt", metadata={"source": "rag.md"})],
        [[1.0, 0.0]],
    )
    chain = make_chain_with_store(store)

    chain.save_index(index_file)

    loaded = SimpleVectorStore()
    loaded.load(str(index_file))
    assert len(loaded) == 1
    assert loaded.chunks[0].metadata == {"source": "rag.md"}
