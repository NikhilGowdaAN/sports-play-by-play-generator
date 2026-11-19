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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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
