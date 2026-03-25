import torch
from rewrite_transformer.tokenizer import BPETokenizer, Vocab
from rewrite_transformer.util import load_dataset
from rewrite_transformer.embedding import TokenEmbedding
from rewrite_transformer.attention import SelfMultiHeadAttention, create_causal_mask
from rewrite_transformer.transformer import Transformer

import time

from rewrite_transformer.util import get_logger

logger = get_logger(__name__)


def test_train_tokenizer():
    logger.info("Testing BPE Tokenizer training...")

    tokenizer = BPETokenizer()

    texts = ["hello world", "how are you"]

    tokenizer.train(texts, 100, max_epoch=100)

    text_ids = tokenizer.encode("fuck you", add_special_tokens=True)
    logger.info(f"Encoded text IDs: {text_ids}")

    text = tokenizer.decode(text_ids)
    logger.info(f"Decoded text: {text}")


def test_load_dataset():
    en_path = "../dataset/TED2020.en-zh_cn.en"
    cn_path = "../dataset/TED2020.en-zh_cn.zh_cn"

    en_data = load_dataset(en_path)
    cn_data = load_dataset(cn_path)

    print(en_data, print(cn_data))


def test_train_bpe():
    start = time.perf_counter()

    en_data = load_dataset("../dataset/TED2020.en-zh_cn.en")
    en_tokenizer = BPETokenizer()
    en_tokenizer.train(en_data, max_vocab_size=32000, max_epoch=32000)
    en_tokenizer.save("en_tokenizer.json")

    cn_data = load_dataset("../dataset/TED2020.en-zh_cn.zh_cn")
    cn_tokenizer = BPETokenizer()
    cn_tokenizer.train(cn_data, max_vocab_size=32000, max_epoch=32000)
    cn_tokenizer.save("cn_tokenizer.json")

    end = time.perf_counter()

    print("耗时：", end - start)


def test_token_embedding():
    tokenizer = BPETokenizer()
    tokenizer.load("./tokenizer.json")

    texts = ["hello world", "fuck you"]
    texts_ids = tokenizer.encode(texts, padding=True)

    print("原始 texts_ids:", texts_ids)

    # Padding: 把短序列补齐到最长序列的长度
    # 使用 PAD token id (假设是 0，或者你可以从 tokenizer 获取)
    # pad_id = tokenizer.vocab.get_id(Vocab.PAD_TOKEN)
    # max_len = max(len(ids) for ids in texts_ids)

    # padded_ids = []
    # for ids in texts_ids:
    #     padded = ids + [pad_id] * (max_len - len(ids))
    #     padded_ids.append(padded)

    # 转换为 tensor: shape = (batch_size, seq_len)
    # input_tensor = torch.tensor(padded_ids)
    # print("Input tensor shape:", input_tensor.shape)

    embedding = TokenEmbedding(tokenizer.vocab_size, 30)
    vecs = embedding.forward(texts_ids)

    # 输出 shape 应该是 (batch_size, seq_len, embed_dim)
    print("Output shape:", vecs.shape)
    print("Embedding vectors:\n", vecs)


def test_attention():
    tokenizer = BPETokenizer()
    tokenizer.load("./tokenizer.json")

    texts = ["hello world", "fuck you"]
    texts_ids = tokenizer.encode(texts, padding=True)
    texts_ids = torch.tensor(texts_ids)

    embedding = TokenEmbedding(tokenizer.vocab_size, 32)
    vecs = embedding.forward(texts_ids)

    attention = SelfMultiHeadAttention(32, 8, 4)
    mask = create_causal_mask(vecs.shape[1], vecs.device)
    output = attention.forward(vecs, mask)

    print("Output shape:", output.shape)
    print("Output vectors:\n", output)


def test_transformer():
    tokenizer = BPETokenizer()
    tokenizer.load("./tokenizer.json")

    texts = ["hello world", "fuck you"]
    texts_ids = tokenizer.encode(texts, padding=True)
    texts_ids = torch.tensor(texts_ids)

    vocab_size = tokenizer.vocab_size

    transformer = Transformer(
        tgt_vocab_size=vocab_size,
        src_vocab_size=vocab_size,
        embed_dim=32,
        head_dim=4,
        head_num=8,
        max_seq_len=32,
        pad_id=tokenizer.vocab.get_id(Vocab.PAD_TOKEN),
    )

    output = transformer.forward(texts_ids, texts_ids)
    print("Output shape:", output.shape)
    print("Output vectors:\n", output)


def test_trainTransformer():
    from rewrite_transformer.train import trainTransformer

    trainTransformer(
        bpe_src_path="../huggingface/en_tokenizer.json",
        bpe_tgt_path="../huggingface/cn_tokenizer.json",
        data_path="../dataset/TED2020.en-zh_cn.en",
        labels_path="../dataset/TED2020.en-zh_cn.zh_cn",
        num_epochs=50,
        use_official_tokenizer=True,
        resume_training=True,
        model_save_path="best_model.pth",
    )


def test_generate():
    from rewrite_transformer.train import generate

    tokenizer_src_path = "../huggingface/en_tokenizer.json"
    tokenizer_tgt_path = "../huggingface/cn_tokenizer.json"
    model_path = "best_model.pth"

    input_texts = [
        "Good morning everyone, and welcome to this presentation.",
        "Today we will discuss the importance of renewable energy.",
        "Artificial intelligence is transforming the world as we know it.",
        "The quick brown fox jumps over the lazy dog.",
        "In a village of La Mancha, the name of which I have no desire to call",
    ]

    outputs = generate(
        bpe_src_path=tokenizer_src_path,
        bpe_tgt_path=tokenizer_tgt_path,
        model_path=model_path,
        texts=input_texts,
        use_official_tokenizer=True,
        model_save_path="best_model.pth",
    )

    for input_text, output_text in zip(input_texts, outputs):
        print(f"Input: {input_text}")
        print(f"Output: {output_text}")
        print("-" * 50)


if __name__ == "__main__":
    # test_train_tokenizer()
    # test_trainTransformer()
    # test_attention()
    # test_transformer()
    test_generate()
