from phase1_chunker import Chunk
from phase1_vector_store import SimpleVectorStore


def test_simple_vector_store_save_load_roundtrip(tmp_path):
    store = SimpleVectorStore()
    chunks = [
        Chunk(content="机器学习是人工智能的一个分支", metadata={"source": "ml.md"}),
        Chunk(content="RAG 会先检索再生成", metadata={"source": "rag.md"}),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    index_file = tmp_path / "simple_index.json"

    store.add(chunks, vectors)
    store.save(str(index_file))

    loaded = SimpleVectorStore()
    loaded.load(str(index_file))

    assert len(loaded) == 2
    assert loaded.vectors == vectors
    assert [chunk.content for chunk in loaded.chunks] == [
        "机器学习是人工智能的一个分支",
        "RAG 会先检索再生成",
    ]
    assert [chunk.metadata for chunk in loaded.chunks] == [
        {"source": "ml.md"},
        {"source": "rag.md"},
    ]


def test_simple_vector_store_clear_empties_data():
    store = SimpleVectorStore()
    store.add(
        [Chunk(content="深度学习使用神经网络", metadata={"source": "dl.md"})],
        [[1.0, 0.0]],
    )

    store.clear()

    assert len(store) == 0
    assert store.search([1.0, 0.0]) == []
