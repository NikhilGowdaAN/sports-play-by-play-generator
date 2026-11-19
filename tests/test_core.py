"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from pathlib import Path
from app.core import (
    milestone_1,
    clean_text,
    get_corpus_stats,
    split_corpus,
    prepare_corpus,
    load_raw_text,
    milestone_2,
    BPETokenizer,
    TokenEmbedding,
    PositionalEmbedding,
    SportsEmbeddings,
)


# Milestone 1: Gather a small corpus of sports commentary or news articles.
def test_milestone_1():
    """Verify that milestone 1 gathers and prepares the sports commentary corpus."""
    corpus = milestone_1()
    assert isinstance(corpus, str), "Corpus should be a string"
    assert len(corpus) > 2000, f"Expected substantial corpus, got {len(corpus)} chars"

    # Verify presence of multi-sport play-by-play and commentary terms
    sports_terms = ["quarter", "touchdown", "buzzer", "fastball", "goal", "commentary", "defense", "rebound"]
    found_terms = [term for term in sports_terms if term in corpus.lower()]
    assert len(found_terms) >= 5, f"Expected diverse sports vocabulary, found {found_terms}"

    # Verify clean text formatting
    assert "\r" not in corpus, "Carriage returns should be normalized"
    assert not corpus.endswith(" "), "Trailing whitespace should be stripped"

    # Verify stats
    stats = get_corpus_stats(corpus)
    assert stats["word_count"] > 300, f"Expected at least 300 words, got {stats['word_count']}"
    assert stats["vocab_size"] > 40, f"Expected rich character vocab, got {stats['vocab_size']}"


def test_clean_text_normalizes_unicode_and_whitespace():
    raw = "‘Winning’ isn’t easy—it’s a “battle”!\r\nLine two  \r\n\r\n\r\n\r\nLine three\u00a0here.\n"
    cleaned = clean_text(raw)
    assert cleaned == "'Winning' isn't easy-it's a \"battle\"!\nLine two\n\nLine three here."


def test_prepare_corpus_with_custom_string():
    custom_text = "Fourth quarter down by three. He shoots... scores!"
    result = prepare_corpus(source=custom_text)
    assert "shoots... scores!" in result
    assert result == custom_text


def test_prepare_corpus_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        prepare_corpus(source="non_existent_sports_file_12345.txt")


def test_split_corpus():
    text = "A" * 100
    train, val = split_corpus(text, train_ratio=0.8)
    assert len(train) == 80
    assert len(val) == 20
    assert train + val == text


def test_cli_corpus_command(capsys):
    from app.cli import main
    main(["corpus", "--preview"])
    captured = capsys.readouterr().out
    assert "Sports Commentary Corpus Statistics:" in captured
    assert "Total Characters:" in captured
    assert "Preview Sample:" in captured


# Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
def test_milestone_2():
    """Verify that milestone 2 trains the BPE tokenizer and creates embeddings."""
    res = milestone_2(vocab_size=300, embed_dim=48)
    assert "tokenizer" in res
    assert "embeddings" in res
    assert "sample_tokens" in res
    assert "sample_embeddings" in res
    assert res["vocab_size"] >= 256
    assert res["embed_dim"] == 48

    # Shape should be (1, seq_len, embed_dim)
    expected_shape = (1, len(res["sample_tokens"]), 48)
    assert res["shape"] == expected_shape

    # Exact round-trip reconstruction for the sample commentary
    decoded = res["tokenizer"].decode(res["sample_tokens"])
    assert decoded == res["sample_text"]


def test_bpe_tokenizer_training_and_merges():
    sports_sample = "Vance drives to the hoop! Vance drives to the hoop again! Goal! Goal!"
    tok = BPETokenizer()
    tok.train(sports_sample, vocab_size=270)
    assert len(tok.merges) > 0, "BPE should learn merges from frequent patterns"
    assert len(tok.vocab) == 256 + len(tok.merges)

    # Encoding should compress repetitive sports text
    raw_bytes_len = len(sports_sample.encode("utf-8"))
    token_len = len(tok.encode(sports_sample))
    assert token_len < raw_bytes_len, f"Expected compression: {token_len} < {raw_bytes_len}"

    # Roundtrip check
    assert tok.decode(tok.encode(sports_sample)) == sports_sample


def test_bpe_tokenizer_zero_oov_and_accents():
    tok = BPETokenizer()
    # Untrained tokenizer has base 256 bytes
    text = "Pelé scored in stoppage time! ⚡ 100% clutch!"
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert decoded == text, "Byte-pair tokenizer must decode arbitrary UTF-8 losslessly"


def test_bpe_tokenizer_special_tokens():
    tok = BPETokenizer()
    eot_id = tok.add_special_token("<|endoftext|>")
    assert eot_id >= 256
    decoded = tok.decode([eot_id])
    assert decoded == "<|endoftext|>"


def test_bpe_tokenizer_serialization(tmp_path):
    tok = BPETokenizer()
    sample = "Fast break tomahawk slam dunk!"
    tok.train(sample, vocab_size=265)
    file_path = tmp_path / "tokenizer.json"
    tok.save(file_path)

    loaded = BPETokenizer.load(file_path)
    assert len(loaded.merges) == len(tok.merges)
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_token_and_positional_embeddings():
    vocab_size = 280
    embed_dim = 32
    max_len = 64

    tok_emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
    pos_emb = PositionalEmbedding(max_len=max_len, embed_dim=embed_dim)
    embeddings = SportsEmbeddings(vocab_size=vocab_size, embed_dim=embed_dim, max_len=max_len)

    tokens = [10, 45, 200, 260]
    out = embeddings(tokens)
    assert out.shape == (1, 4, 32)
    # Check multi-index indexing and vector length
    vec = out[0, 0]
    assert len(vec) == 32

    # Verify sequence exceeding max_len raises ValueError
    with pytest.raises(ValueError):
        embeddings(list(range(70)))


def test_cli_tokenize_and_embed_commands(capsys):
    from app.cli import main
    main(["tokenize", "Marcus Vance for three!"])
    out = capsys.readouterr().out
    assert "Token count:" in out
    assert "Token IDs:" in out

    main(["embed", "Clutch buzzer-beater!"])
    out2 = capsys.readouterr().out
    assert "Embedding shape:" in out2
    assert "Embedding vector dimension: 64" in out2


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Train the GPT model using the sports text data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
