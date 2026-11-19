"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import random
import re
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# ==============================================================================
# --- Milestone 1: Gather a small corpus of sports commentary or news articles.
# Hint: Raw text input preparation.
# ==============================================================================

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "sports_corpus.txt"

FALLBACK_SPORTS_SNIPPETS = """=== BASKETBALL COMMENTARY ===
Thirty seconds left on the clock in the fourth quarter. Down by one, Vance dribbles at the top of the key. He drives past his defender, elevates at the rim, and throws down a thunderous dunk as time expires! The arena erupts in celebration!

=== SOCCER COMMENTARY ===
In the final minute of stoppage time, a curling free kick is whipped into the six-yard box. The striker rises above the defense and buries a bullet header into the top corner! An unbelievable equalizer!

=== FOOTBALL COMMENTARY ===
Fourth and goal with five seconds remaining. The quarterback scrambles out of the pocket, avoids the blitz, and fires a laser into the back of the end zone for the game-winning touchdown!
"""


def clean_text(raw_text: str) -> str:
    """Normalize and clean raw text input for downstream processing."""
    if not raw_text:
        return ""
    # Normalize unicode quotation marks, dashes, and spaces
    text = (
        raw_text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u00a0", " ")
    )
    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_raw_text(file_path: str | Path) -> str:
    """Load raw text from a file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return path.read_text(encoding="utf-8")


def get_corpus_stats(text: str) -> dict[str, int]:
    """Calculate statistics for the prepared corpus."""
    words = re.findall(r"\b\w+\b", text)
    unique_words = set(w.lower() for w in words)
    non_empty_lines = [l for l in text.splitlines() if l.strip()]
    return {
        "char_count": len(text),
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "line_count": len(non_empty_lines),
        "vocab_size": len(set(text)),
    }


def split_corpus(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split corpus text into training and validation splits."""
    if not text:
        return "", ""
    split_idx = int(len(text) * train_ratio)
    return text[:split_idx], text[split_idx:]


def prepare_corpus(source: str | Path | None = None, clean: bool = True) -> str:
    """Prepare and normalize the sports text corpus from a file or text source.

    If source is None, attempts to load from DEFAULT_CORPUS_PATH or fallback snippets.
    """
    raw_text = ""
    if source is not None:
        if isinstance(source, Path):
            raw_text = load_raw_text(source)
        elif isinstance(source, str):
            path = Path(source)
            if path.is_file():
                raw_text = load_raw_text(path)
            elif "\n" in source or (" " in source and not any(source.endswith(ext) for ext in [".txt", ".text", ".md", ".json"])):
                raw_text = source
            else:
                raw_text = load_raw_text(path)
    else:
        if DEFAULT_CORPUS_PATH.exists():
            raw_text = load_raw_text(DEFAULT_CORPUS_PATH)
        else:
            raw_text = FALLBACK_SPORTS_SNIPPETS

    if clean:
        raw_text = clean_text(raw_text)

    if not raw_text:
        raise ValueError("Corpus text cannot be empty after preparation.")

    return raw_text


def milestone_1(corpus_path: str | Path | None = None) -> str:
    """Milestone 1: Gather a small corpus of sports commentary or news articles.

    Loads and prepares the raw sports commentary text corpus, ensuring it is
    cleaned and ready for tokenization and model training.
    """
    return prepare_corpus(source=corpus_path, clean=True)


# ==============================================================================
# --- Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
# Hint: Byte-pair tokenizer and embeddings.
# ==============================================================================

def milestone_2(corpus_text: str | None = None, vocab_size: int = 320, embed_dim: int = 64) -> dict:
    """Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.

    1. Gathers or loads the sports commentary text corpus.
    2. Trains a Byte-Pair Encoding (BPE) tokenizer on the sports corpus.
    3. Builds Token and Positional Embeddings for the learned vocabulary.
    4. Encodes and generates embeddings for a sample play-by-play commentary scenario.
    """
    if corpus_text is None:
        corpus_text = milestone_1()

    # 1. Initialize and train BPE tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_text, vocab_size=vocab_size)

    # 2. Instantiate Sports Embeddings (Token Embedding + Positional Embedding)
    actual_vocab_size = len(tokenizer.vocab)
    embeddings = SportsEmbeddings(vocab_size=actual_vocab_size, embed_dim=embed_dim)

    # 3. Generate embeddings for a sample commentary snippet
    sample_text = "Marcus Vance hits the step-back jumper at the buzzer to win the championship!"
    sample_tokens = tokenizer.encode(sample_text)
    sample_embeddings = embeddings(sample_tokens)

    return {
        "tokenizer": tokenizer,
        "embeddings": embeddings,
        "sample_text": sample_text,
        "sample_tokens": sample_tokens,
        "sample_embeddings": sample_embeddings,
        "vocab_size": actual_vocab_size,
        "embed_dim": embed_dim,
        "shape": sample_embeddings.shape,
    }


class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer built from scratch.

    Starts with a 256-byte vocabulary (0..255) for complete UTF-8 coverage with zero OOV tokens,
    then learns frequent adjacent byte-pair merges from sports text.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        """Count frequency of all adjacent token pairs."""
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all consecutive occurrences of pair with new_id in tokens."""
        new_tokens = []
        i = 0
        n = len(tokens)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str, vocab_size: int = 320) -> None:
        """Learn BPE merges from training text up to target vocab_size."""
        num_merges = vocab_size - 256
        if num_merges <= 0:
            return

        tokens = list(text.encode("utf-8"))
        for _ in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # Stop if no pair occurs more than once
                break
            new_id = 256 + len(self.merges)
            tokens = self.merge_tokens(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def add_special_token(self, token: str) -> int:
        """Register a special token string (e.g. '<|endoftext|>')."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        new_id = 256 + len(self.merges) + len(self.special_tokens)
        self.special_tokens[token] = new_id
        self.inverse_special_tokens[new_id] = token
        return new_id

    def encode(self, text: str) -> list[int]:
        """Encode text string into token IDs using learned BPE merges."""
        if not text:
            return []
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Find the pair that was merged earliest during training
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            tokens = self.merge_tokens(tokens, pair, new_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Losslessly decode a sequence of token IDs back into a UTF-8 string."""
        byte_chunks = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            elif tid in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[tid].encode("utf-8"))
            else:
                byte_chunks.append(b"")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        """Export tokenizer merges and special tokens as a serializable dict."""
        return {
            "merges": [[p[0], p[1], nid] for p, nid in self.merges.items()],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        """Reconstruct tokenizer from a serializable dict."""
        tok = cls()
        for p0, p1, nid in data.get("merges", []):
            pair = (p0, p1)
            tok.merges[pair] = nid
            tok.vocab[nid] = tok.vocab[p0] + tok.vocab[p1]
        for token, tid in data.get("special_tokens", {}).items():
            tok.special_tokens[token] = tid
            tok.inverse_special_tokens[tid] = token
        return tok

    def save(self, file_path: str | Path) -> None:
        """Save tokenizer configuration to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: str | Path) -> "BPETokenizer":
        """Load tokenizer from a JSON file."""
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class SimpleTensor:
    """A lightweight multi-dimensional tensor container for environments without PyTorch."""

    def __init__(self, data: list):
        self.data = data
        self.shape = self._get_shape(data)

    def _get_shape(self, data) -> tuple[int, ...]:
        if not isinstance(data, list):
            return ()
        if len(data) == 0:
            return (0,)
        return (len(data),) + self._get_shape(data[0])

    def tolist(self) -> list:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            curr = self.data
            for idx in item:
                curr = curr[idx]
            if isinstance(curr, list):
                return SimpleTensor(curr)
            return curr
        res = self.data[item]
        if isinstance(res, list):
            return SimpleTensor(res)
        return res

    def __add__(self, other):
        other_data = other.data if isinstance(other, SimpleTensor) else other

        def _add(a, b):
            if isinstance(a, list) and isinstance(b, list):
                if len(a) == len(b):
                    return [_add(x, y) for x, y in zip(a, b)]
                if len(a) == 1:
                    return [_add(a[0], y) for y in b]
                if len(b) == 1:
                    return [_add(x, b[0]) for x in a]
                raise ValueError(f"Shape mismatch: {len(a)} vs {len(b)}")
            return a + b

        return SimpleTensor(_add(self.data, other_data))

    def squeeze(self, dim: int = 0):
        if dim == 0 and len(self.shape) > 0 and self.shape[0] == 1:
            return SimpleTensor(self.data[0])
        return self

    def unsqueeze(self, dim: int = 0):
        if dim == 0:
            return SimpleTensor([self.data])
        return self

    def __repr__(self) -> str:
        return f"SimpleTensor(shape={self.shape})"


if HAS_TORCH:
    class TokenEmbedding(nn.Module):
        """Token embedding layer mapping token IDs to dense vectors using PyTorch."""

        def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
            super().__init__()
            torch.manual_seed(seed)
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        def forward(self, idx):
            if isinstance(idx, list):
                idx = torch.tensor(idx, dtype=torch.long)
            return self.embedding(idx)

    class PositionalEmbedding(nn.Module):
        """Learned positional embedding layer mapping sequence positions using PyTorch."""

        def __init__(self, max_len: int, embed_dim: int, seed: int = 42):
            super().__init__()
            torch.manual_seed(seed + 1)
            self.max_len = max_len
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(max_len, embed_dim)

        def forward(self, seq_len: int):
            positions = torch.arange(0, seq_len, dtype=torch.long)
            return self.embedding(positions).unsqueeze(0)

    class SportsEmbeddings(nn.Module):
        """Combined Token and Positional Embeddings for sports text (PyTorch)."""

        def __init__(self, vocab_size: int, embed_dim: int = 64, max_len: int = 512, seed: int = 42):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.max_len = max_len
            self.tok_emb = TokenEmbedding(vocab_size, embed_dim, seed=seed)
            self.pos_emb = PositionalEmbedding(max_len, embed_dim, seed=seed)

        def forward(self, idx):
            if isinstance(idx, list):
                if idx and isinstance(idx[0], int):
                    idx = [idx]
                idx = torch.tensor(idx, dtype=torch.long)
            elif isinstance(idx, torch.Tensor) and idx.ndim == 1:
                idx = idx.unsqueeze(0)
            B, T = idx.shape
            tok_vecs = self.tok_emb(idx)
            pos_vecs = self.pos_emb(T)
            return tok_vecs + pos_vecs

else:
    class TokenEmbedding:
        """Token embedding layer mapping token IDs to dense vectors."""

        def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
            rng = random.Random(seed)
            std = 0.02
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.weight = [[rng.gauss(0, std) for _ in range(embed_dim)] for _ in range(vocab_size)]

        def __call__(self, idx: list[int] | list[list[int]]) -> SimpleTensor:
            if not idx:
                return SimpleTensor([])
            if isinstance(idx[0], int):
                seq_emb = [self.weight[tid] for tid in idx]
                return SimpleTensor([seq_emb])
            elif isinstance(idx[0], list):
                batch_emb = [[self.weight[tid] for tid in seq] for seq in idx]
                return SimpleTensor(batch_emb)
            raise TypeError("idx must be list of ints or list of list of ints")

    class PositionalEmbedding:
        """Learned positional embedding layer mapping sequence positions."""

        def __init__(self, max_len: int, embed_dim: int, seed: int = 42):
            rng = random.Random(seed + 1)
            std = 0.02
            self.max_len = max_len
            self.embed_dim = embed_dim
            self.weight = [[rng.gauss(0, std) for _ in range(embed_dim)] for _ in range(max_len)]

        def __call__(self, seq_len: int) -> SimpleTensor:
            if seq_len > self.max_len:
                raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
            pos_emb = [self.weight[p] for p in range(seq_len)]
            return SimpleTensor([pos_emb])

    class SportsEmbeddings:
        """Combined Token and Positional Embeddings for sports commentary text."""

        def __init__(self, vocab_size: int, embed_dim: int = 64, max_len: int = 512, seed: int = 42):
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.max_len = max_len
            self.tok_emb = TokenEmbedding(vocab_size, embed_dim, seed=seed)
            self.pos_emb = PositionalEmbedding(max_len, embed_dim, seed=seed)

        def forward(self, token_ids: list[int] | list[list[int]]) -> SimpleTensor:
            if not token_ids:
                return SimpleTensor([])
            if isinstance(token_ids[0], int):
                seq_len = len(token_ids)
                toks = self.tok_emb(token_ids)
                poss = self.pos_emb(seq_len)
                return toks + poss
            else:
                seq_len = len(token_ids[0])
                toks = self.tok_emb(token_ids)
                poss = self.pos_emb(seq_len)
                return toks + poss

        def __call__(self, token_ids) -> SimpleTensor:
            return self.forward(token_ids)


# ==============================================================================
# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal self-attention and transformer block assembly.
# ==============================================================================
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# ==============================================================================
# --- Milestone 4: Train the GPT model using the sports text data.
# Hint: Model training.
# ==============================================================================
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# ==============================================================================
# --- Milestone 5: Develop a text generation function to produce short commentary snippets.
# Hint: Text generation.
# ==============================================================================
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# ==============================================================================
# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
# Hint: Applying text generation in a CLI.
# ==============================================================================
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
