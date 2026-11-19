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
    milestone_3,
    CausalSelfAttention,
    MLP,
    LayerNorm,
    TransformerBlock,
    SimpleTensor,
    milestone_4,
    SportsGPT,
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
    with pytest.raises((ValueError, IndexError)):
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
def test_milestone_3():
    """Verify causal self-attention and transformer block forward pass."""
    res = milestone_3(embed_dim=32, num_heads=4)
    assert "block" in res
    assert "attention" in res
    assert "input_embeddings" in res
    assert "block_output" in res
    assert res["causal_mask_verified"] is True
    assert res["residual_verified"] is True
    assert res["input_shape"] == res["output_shape"]
    assert res["output_shape"][-1] == 32


def test_causal_self_attention_strict_masking():
    """Verify that upper-triangular weights in causal self-attention are strictly zero."""
    d_model = 32
    num_heads = 4
    T = 6
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads)
    # Mock sequence: (1, T, d_model)
    dummy_input = SimpleTensor([[[float(i + k) for k in range(d_model)] for i in range(T)]])
    out, weights = attn.forward(dummy_input, return_weights=True)

    assert out.shape == (1, T, d_model)
    w_list = weights.tolist() if hasattr(weights, "tolist") else weights
    # For every head, for every row i, future columns j > i must have weight 0.0
    for head_w in w_list[0]:
        for i in range(T):
            for j in range(T):
                if j > i:
                    assert head_w[i][j] == 0.0, f"Future leakage detected at ({i}, {j})"
            # Lower triangular weights must sum to 1.0 (valid probability distribution)
            row_sum = sum(head_w[i][:i + 1])
            assert abs(row_sum - 1.0) < 1e-5, f"Softmax row {i} does not sum to 1.0: {row_sum}"


def test_causality_future_token_invariance():
    """Verify that changing a future token has zero effect on earlier token outputs."""
    d_model = 16
    num_heads = 2
    T = 4
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads)

    seq_a = [[1.0] * d_model, [2.0] * d_model, [3.0] * d_model, [4.0] * d_model]
    seq_b = [[1.0] * d_model, [2.0] * d_model, [3.0] * d_model, [999.0] * d_model]  # change only last token

    out_a = attn.forward(SimpleTensor([seq_a]))
    out_b = attn.forward(SimpleTensor([seq_b]))

    # Positions 0, 1, 2 must be identical
    for pos in range(3):
        for d in range(d_model):
            diff = abs(out_a[0, pos, d] - out_b[0, pos, d])
            assert diff < 1e-6, f"Position {pos} leaked future information: diff={diff}"

    # Position 3 must be different
    diff_last = sum(abs(out_a[0, 3, d] - out_b[0, 3, d]) for d in range(d_model))
    assert diff_last > 1.0, "Position 3 was expected to change"


def test_transformer_block_forward_and_residual():
    """Verify transformer block executes Pre-LN attention + MLP and preserves dimensionality."""
    d_model = 32
    num_heads = 4
    block = TransformerBlock(d_model=d_model, num_heads=num_heads)

    dummy_input = SimpleTensor([[[float(t + d * 0.1) for d in range(d_model)] for t in range(8)]])
    out = block.forward(dummy_input)

    assert out.shape == (1, 8, d_model)
    # Ensure transformation took place (output differs from raw input)
    diff = sum(abs(out[0, 0, d] - dummy_input[0, 0, d]) for d in range(d_model))
    assert diff > 0.01, "Transformer block should perform non-trivial transformation"


def test_layer_norm():
    """Verify LayerNorm normalizes mean to 0 and variance to 1."""
    d_model = 16
    ln = LayerNorm(d_model=d_model)
    raw_vec = [float(i * 3 + 5) for i in range(d_model)]
    normed = ln.forward(SimpleTensor([[raw_vec]]))

    mean = sum(normed[0, 0, d] for d in range(d_model)) / d_model
    var = sum((normed[0, 0, d] - mean) ** 2 for d in range(d_model)) / d_model

    assert abs(mean) < 1e-4, f"LayerNorm mean should be ~0, got {mean}"
    assert abs(var - 1.0) < 1e-3, f"LayerNorm variance should be ~1, got {var}"


def test_cli_transformer_command(capsys):
    from app.cli import main
    main(["transformer", "Vance drives to the lane!"])
    out = capsys.readouterr().out
    assert "Transformer Block Forward Pass:" in out
    assert "Output shape:" in out
    assert "Causal masking:    PASSED" in out
    assert "Residual check:    PASSED" in out


# Milestone 4: Train the GPT model using the sports text data.
def test_milestone_4():
    """Verify that milestone 4 trains SportsGPT and reduces cross-entropy loss."""
    res = milestone_4(steps=15, embed_dim=32, num_heads=4, lr=0.08)
    assert "model" in res
    assert "tokenizer" in res
    assert "losses" in res
    assert len(res["losses"]) == 15
    assert res["initial_loss"] > 0
    assert res["final_loss"] > 0
    assert bool(res["training_verified"]) is True
    assert res["final_loss"] < res["initial_loss"], f"Expected loss drop: {res['final_loss']} < {res['initial_loss']}"
    assert res["loss_reduction"] > 0.5, f"Expected noticeable reduction, got {res['loss_reduction']}"


def test_sports_gpt_forward_shape_and_loss():
    """Verify SportsGPT forward pass output shape and loss computation."""
    vocab_size = 280
    embed_dim = 32
    model = SportsGPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=4, num_layers=1, max_len=64)

    # Forward without targets
    tokens = [1, 5, 20, 100, 250, 279]
    logits = model.forward(tokens)
    assert logits.shape == (1, 6, vocab_size)

    # Forward with targets
    targets = [5, 20, 100, 250, 279, 1]
    logits_with_loss, loss = model.forward(tokens, targets=targets)
    assert logits_with_loss.shape == (1, 6, vocab_size)
    assert loss > 0.0, f"Cross-entropy loss should be positive, got {loss}"


def test_sports_gpt_training_decreases_loss():
    """Verify that train_step updates model parameters and decreases cross-entropy loss."""
    vocab_size = 270
    embed_dim = 32
    model = SportsGPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=2, num_layers=1, max_len=32)

    x = [2, 10, 15, 80, 200, 256, 12, 45]
    y = [10, 15, 80, 200, 256, 12, 45, 2]

    _, initial_loss = model.forward(x, targets=y)
    for _ in range(12):
        model.train_step(x, y, lr=0.1)
    _, final_loss = model.forward(x, targets=y)

    assert final_loss < initial_loss, f"Loss should decrease: {final_loss} < {initial_loss}"
    assert (initial_loss - final_loss) > 1.0, f"Expected significant loss reduction: {initial_loss - final_loss}"


def test_sports_gpt_save_and_load(tmp_path):
    """Verify SportsGPT weights serialization and deserialization."""
    vocab_size = 265
    embed_dim = 32
    model = SportsGPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=2, num_layers=1, max_len=32)

    file_path = tmp_path / "sports_gpt_model.json"
    model.save(file_path)

    loaded_model = SportsGPT.load(file_path)
    sample_tokens = [1, 2, 3, 4]
    out_orig = model.forward(sample_tokens)
    out_loaded = loaded_model.forward(sample_tokens)

    # Assert outputs are identical
    diff = sum(abs(out_orig[0, 0, v] - out_loaded[0, 0, v]) for v in range(vocab_size))
    assert diff < 1e-6, f"Loaded model output mismatch: diff={diff}"


def test_cli_train_command(capsys):
    from app.cli import main
    main(["train", "--steps", "5"])
    out = capsys.readouterr().out
    assert "Model Training on Sports Corpus:" in out
    assert "Initial Loss:" in out
    assert "Final Loss:" in out
    assert "PASSED" in out


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
