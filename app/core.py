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


class MilestoneResult(dict):
    """Dictionary subclass supporting attribute access (e.g. res.tokenizer or res['tokenizer'])."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'MilestoneResult' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


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

    return MilestoneResult({
        "tokenizer": tokenizer,
        "embeddings": embeddings,
        "sample_text": sample_text,
        "sample_tokens": sample_tokens,
        "sample_embeddings": sample_embeddings,
        "vocab_size": actual_vocab_size,
        "embed_dim": embed_dim,
        "shape": sample_embeddings.shape,
    })


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
            if isinstance(idx, SimpleTensor):
                idx = idx.data
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
            if seq_len > self.max_len:
                raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
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
            if isinstance(idx, SimpleTensor):
                idx = idx.data
            if isinstance(idx, list):
                if idx and isinstance(idx[0], int):
                    idx = [idx]
                idx = torch.tensor(idx, dtype=torch.long)
            elif isinstance(idx, torch.Tensor) and idx.ndim == 1:
                idx = idx.unsqueeze(0)
            B, T = idx.shape
            if T > self.max_len:
                raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
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

def milestone_3(embeddings_input=None, embed_dim: int = 64, num_heads: int = 4) -> dict:
    """Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.

    1. Loads sports commentary embeddings from Milestone 2 (or uses provided embeddings).
    2. Constructs CausalSelfAttention with strict upper-triangular masking.
    3. Assembles the TransformerBlock (Pre-LN LayerNorm + Multi-Head Attention + MLP + Residuals).
    4. Computes forward pass over sports embeddings and verifies causal attention properties.
    """
    if embeddings_input is None:
        m2 = milestone_2(embed_dim=embed_dim)
        embeddings_input = m2["sample_embeddings"]

    # Instantiate attention mechanism and transformer block
    attention = CausalSelfAttention(d_model=embed_dim, num_heads=num_heads)
    block = TransformerBlock(d_model=embed_dim, num_heads=num_heads)

    # 1. Forward pass through Causal Self-Attention
    attn_out, attn_weights = attention.forward(embeddings_input, return_weights=True)

    # 2. Forward pass through complete Transformer Block
    block_out = block.forward(embeddings_input)

    # Verify causal masking: future positions j > i must have exactly 0.0 attention weight
    causal_mask_verified = True
    w_list = attn_weights.tolist() if hasattr(attn_weights, "tolist") else attn_weights
    for head_w in w_list[0]:
        T = len(head_w)
        for i in range(T):
            for j in range(i + 1, T):
                if head_w[i][j] > 1e-6:
                    causal_mask_verified = False
                    break

    input_shape = getattr(embeddings_input, "shape", (len(embeddings_input), len(embeddings_input[0]), len(embeddings_input[0][0])))
    output_shape = getattr(block_out, "shape", (len(block_out), len(block_out[0]), len(block_out[0][0])))

    return MilestoneResult({
        "block": block,
        "attention": attention,
        "input_embeddings": embeddings_input,
        "attention_output": attn_out,
        "attention_weights": attn_weights,
        "block_output": block_out,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "causal_mask_verified": causal_mask_verified,
        "residual_verified": (input_shape == output_shape),
    })


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (GELU) activation function."""
    import math
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def matmul_2d(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply 2D matrices A (N, K) and B (K, M) -> (N, M)."""
    N = len(A)
    K = len(A[0])
    M = len(B[0])
    out = [[0.0] * M for _ in range(N)]
    for i in range(N):
        for k in range(K):
            a_ik = A[i][k]
            if a_ik == 0.0:
                continue
            for j in range(M):
                out[i][j] += a_ik * B[k][j]
    return out


def softmax_1d(row: list[float]) -> list[float]:
    """Numerically stable softmax over a 1D vector."""
    import math
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(row)] * len(row)


if HAS_TORCH:
    class LayerNorm(nn.Module):
        """Layer Normalization layer (PyTorch)."""

        def __init__(self, d_model: int, eps: float = 1e-5):
            super().__init__()
            self.d_model = d_model
            self.ln = nn.LayerNorm(d_model, eps=eps)

        def forward(self, x):
            if isinstance(x, SimpleTensor):
                x = torch.tensor(x.data, dtype=torch.float32)
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.float32)
            return self.ln(x)

        def __call__(self, x):
            return self.forward(x)

    class CausalSelfAttention(nn.Module):
        """Multi-head Causal Self-Attention mechanism (PyTorch)."""

        def __init__(self, d_model: int = 64, num_heads: int = 4, max_len: int = 512, seed: int = 42):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            torch.manual_seed(seed)
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
            self.register_buffer("bias", mask)

        def forward(self, x, return_weights: bool = False):
            import math
            if isinstance(x, SimpleTensor):
                x = torch.tensor(x.data, dtype=torch.float32)
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 2:
                x = x.unsqueeze(0)

            B, T, C = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.d_model, dim=2)

            k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            weights = torch.softmax(att, dim=-1)
            y = weights @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            out = self.c_proj(y)
            if return_weights:
                return out, weights
            return out

        def __call__(self, x, return_weights: bool = False):
            return self.forward(x, return_weights=return_weights)

    class MLP(nn.Module):
        """Feed-Forward Network with GELU activation (PyTorch)."""

        def __init__(self, d_model: int = 64, expansion: int = 4, seed: int = 42):
            super().__init__()
            torch.manual_seed(seed + 10)
            self.c_fc = nn.Linear(d_model, expansion * d_model)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(expansion * d_model, d_model)

        def forward(self, x):
            if isinstance(x, SimpleTensor):
                x = torch.tensor(x.data, dtype=torch.float32)
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.float32)
            return self.c_proj(self.gelu(self.c_fc(x)))

        def __call__(self, x):
            return self.forward(x)

    class TransformerBlock(nn.Module):
        """Pre-LN Transformer Block containing Causal Attention and MLP (PyTorch)."""

        def __init__(self, d_model: int = 64, num_heads: int = 4, max_len: int = 512, seed: int = 42):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.ln_1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, num_heads, max_len=max_len, seed=seed)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = MLP(d_model, seed=seed)

        def forward(self, x):
            if isinstance(x, SimpleTensor):
                x = torch.tensor(x.data, dtype=torch.float32)
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.float32)
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

        def __call__(self, x):
            return self.forward(x)

else:
    class LayerNorm:
        """Layer Normalization across the feature dimension."""

        def __init__(self, d_model: int, eps: float = 1e-5):
            self.d_model = d_model
            self.eps = eps
            self.gamma = [1.0] * d_model
            self.beta = [0.0] * d_model

        def forward(self, x_3d: list | SimpleTensor) -> SimpleTensor:
            import math
            raw_x = x_3d.data if isinstance(x_3d, SimpleTensor) else x_3d
            out = []
            for batch in raw_x:
                b_out = []
                for vec in batch:
                    mean = sum(vec) / self.d_model
                    var = sum((v - mean) ** 2 for v in vec) / self.d_model
                    std = math.sqrt(var + self.eps)
                    normed = [(v - mean) / std * g + b for v, g, b in zip(vec, self.gamma, self.beta)]
                    b_out.append(normed)
                out.append(b_out)
            return SimpleTensor(out)

        def __call__(self, x_3d) -> SimpleTensor:
            return self.forward(x_3d)

    class CausalSelfAttention:
        """Multi-head Causal Self-Attention mechanism built from scratch."""

        def __init__(self, d_model: int = 64, num_heads: int = 4, max_len: int = 512, seed: int = 42):
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.max_len = max_len

            rng = random.Random(seed)
            std = 0.02
            self.w_q = [[rng.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
            self.w_k = [[rng.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
            self.w_v = [[rng.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
            self.w_o = [[rng.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]

        def forward(self, x: list | SimpleTensor, return_weights: bool = False):
            import math
            raw_x = x.data if isinstance(x, SimpleTensor) else x
            if not isinstance(raw_x[0], list):
                raw_x = [raw_x]
            if not isinstance(raw_x[0][0], list):
                raw_x = [raw_x]

            B = len(raw_x)
            T = len(raw_x[0])
            D = self.d_model
            H = self.num_heads
            d_k = self.d_k
            scale = 1.0 / math.sqrt(d_k)

            all_heads_out = []
            all_head_weights = []

            for b in range(B):
                seq = raw_x[b]  # (T, D)
                Q = matmul_2d(seq, self.w_q)
                K = matmul_2d(seq, self.w_k)
                V = matmul_2d(seq, self.w_v)

                batch_attn_out = [[0.0] * D for _ in range(T)]
                b_weights = []

                for h in range(H):
                    start = h * d_k
                    end = start + d_k
                    Q_h = [row[start:end] for row in Q]
                    K_h = [row[start:end] for row in K]
                    V_h = [row[start:end] for row in V]

                    head_w = []
                    for i in range(T):
                        row_scores = []
                        q_i = Q_h[i]
                        for j in range(T):
                            if j > i:
                                row_scores.append(-1e9)  # Causal mask: no future information
                            else:
                                dot = sum(q_i[k] * K_h[j][k] for k in range(d_k))
                                row_scores.append(dot * scale)

                        weights_i = softmax_1d(row_scores)
                        head_w.append(weights_i)

                        # Weighted sum over values
                        v_out_i = [0.0] * d_k
                        for j in range(i + 1):
                            w = weights_i[j]
                            v_j = V_h[j]
                            for k in range(d_k):
                                v_out_i[k] += w * v_j[k]

                        for k in range(d_k):
                            batch_attn_out[i][start + k] = v_out_i[k]

                    b_weights.append(head_w)

                projected = matmul_2d(batch_attn_out, self.w_o)
                all_heads_out.append(projected)
                all_head_weights.append(b_weights)

            out_tensor = SimpleTensor(all_heads_out)
            weights_tensor = SimpleTensor(all_head_weights)
            if return_weights:
                return out_tensor, weights_tensor
            return out_tensor

        def __call__(self, x, return_weights: bool = False):
            return self.forward(x, return_weights=return_weights)

    class MLP:
        """Position-wise Feed-Forward Network with GELU activation."""

        def __init__(self, d_model: int = 64, expansion: int = 4, seed: int = 42):
            self.d_model = d_model
            d_ff = d_model * expansion
            rng = random.Random(seed + 10)
            std = 0.02
            self.w1 = [[rng.gauss(0, std) for _ in range(d_ff)] for _ in range(d_model)]
            self.b1 = [0.0] * d_ff
            self.w2 = [[rng.gauss(0, std) for _ in range(d_model)] for _ in range(d_ff)]
            self.b2 = [0.0] * d_model

        def forward(self, x_3d: list | SimpleTensor) -> SimpleTensor:
            raw_x = x_3d.data if isinstance(x_3d, SimpleTensor) else x_3d
            out = []
            for batch in raw_x:
                h1 = matmul_2d(batch, self.w1)
                for row in h1:
                    for j in range(len(row)):
                        row[j] = gelu(row[j] + self.b1[j])
                h2 = matmul_2d(h1, self.w2)
                for row in h2:
                    for j in range(len(row)):
                        row[j] += self.b2[j]
                out.append(h2)
            return SimpleTensor(out)

        def __call__(self, x_3d) -> SimpleTensor:
            return self.forward(x_3d)

    class TransformerBlock:
        """Assembled Pre-LN Transformer Block combining Causal Self-Attention and MLP."""

        def __init__(self, d_model: int = 64, num_heads: int = 4, max_len: int = 512, seed: int = 42):
            self.d_model = d_model
            self.num_heads = num_heads
            self.ln1 = LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, num_heads=num_heads, max_len=max_len, seed=seed)
            self.ln2 = LayerNorm(d_model)
            self.mlp = MLP(d_model, expansion=4, seed=seed)

        def forward(self, x: list | SimpleTensor) -> SimpleTensor:
            raw_x = x.data if isinstance(x, SimpleTensor) else x
            # 1. Pre-LN Attention residual connection
            norm1 = self.ln1(raw_x)
            attn_out = self.attn(norm1)
            raw_attn = attn_out.data if isinstance(attn_out, SimpleTensor) else attn_out

            B = len(raw_x)
            T = len(raw_x[0])
            D = self.d_model
            h1 = [[[raw_x[b][t][d] + raw_attn[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]

            # 2. Pre-LN MLP residual connection
            norm2 = self.ln2(h1)
            mlp_out = self.mlp(norm2)
            raw_mlp = mlp_out.data if isinstance(mlp_out, SimpleTensor) else mlp_out

            h2 = [[[h1[b][t][d] + raw_mlp[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]
            return SimpleTensor(h2)

        def __call__(self, x) -> SimpleTensor:
            return self.forward(x)


# ==============================================================================
# --- Milestone 4: Train the GPT model using the sports text data.
# Hint: Model training.
# ==============================================================================

def milestone_4(corpus_text: str | None = None, steps: int = 20, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 1, lr: float = 0.08) -> dict:
    """Milestone 4: Train the GPT model using the sports text data.

    1. Loads or accepts the sports commentary corpus.
    2. Trains the BPE tokenizer to build vocabulary.
    3. Encodes the sports text into training sequences.
    4. Initializes the full SportsGPT model (Embeddings + Transformer Blocks + LM Head).
    5. Trains the model using autoregressive next-token prediction and cross-entropy loss.
    6. Verifies that the training loss significantly decreases.
    """
    if corpus_text is None:
        corpus_text = milestone_1()

    # 1. Train or load BPE tokenizer on corpus
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_text, vocab_size=300)
    actual_vocab_size = len(tokenizer.vocab)

    # 2. Encode corpus to tokens
    tokens = tokenizer.encode(corpus_text)

    # 3. Instantiate full SportsGPT model
    model = SportsGPT(
        vocab_size=actual_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=128,
        seed=42,
    )

    # 4. Prepare training sequences (predict next token)
    block_size = min(32, len(tokens) - 1)
    x_train = tokens[:block_size]
    y_train = tokens[1:block_size + 1]

    # Evaluate initial loss before training
    _, initial_loss = model.forward(x_train, targets=y_train)

    # Train model across steps
    losses = []
    for step in range(steps):
        loss_val = model.train_step(x_train, y_train, lr=lr)
        losses.append(loss_val)

    # Evaluate final loss on reference chunk
    _, final_loss = model.forward(x_train, targets=y_train)

    return MilestoneResult({
        "model": model,
        "tokenizer": tokenizer,
        "initial_loss": float(initial_loss.detach() if hasattr(initial_loss, "detach") else initial_loss),
        "final_loss": float(final_loss.detach() if hasattr(final_loss, "detach") else final_loss),
        "loss_reduction": float((initial_loss - final_loss).detach() if hasattr(initial_loss, "detach") else (initial_loss - final_loss)),
        "losses": [float(l) for l in losses],
        "trained_steps": steps,
        "vocab_size": actual_vocab_size,
        "embed_dim": embed_dim,
        "training_verified": bool(final_loss < initial_loss),
    })


if HAS_TORCH:
    class SportsGPT(nn.Module):
        """Full GPT model for sports commentary generation (PyTorch)."""

        def __init__(self, vocab_size: int = 300, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, max_len: int = 128, seed: int = 42):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.max_len = max_len

            torch.manual_seed(seed)
            self.embeddings = SportsEmbeddings(vocab_size, embed_dim, max_len=max_len, seed=seed)
            self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads=num_heads, max_len=max_len, seed=seed + i) for i in range(num_layers)])
            self.ln_f = nn.LayerNorm(embed_dim)
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=True)
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        def forward(self, idx, targets=None):
            if isinstance(idx, SimpleTensor):
                idx = idx.data
            if isinstance(idx, list):
                if idx and isinstance(idx[0], int):
                    idx = [idx]
                idx = torch.tensor(idx, dtype=torch.long)
            elif isinstance(idx, torch.Tensor) and idx.ndim == 1:
                idx = idx.unsqueeze(0)

            x = self.embeddings(idx)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            if targets is None:
                return logits

            if isinstance(targets, SimpleTensor):
                targets = targets.data
            if isinstance(targets, list):
                if targets and isinstance(targets[0], int):
                    targets = [targets]
                targets = torch.tensor(targets, dtype=torch.long)
            elif isinstance(targets, torch.Tensor) and targets.ndim == 1:
                targets = targets.unsqueeze(0)

            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

        def train_step(self, x_tokens, y_targets, lr=1e-3):
            self.train()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.zero_grad()
            logits, loss = self.forward(x_tokens, targets=y_targets)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def to_dict(self) -> dict:
            return {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_len": self.max_len,
                "state_dict": {k: v.cpu().tolist() for k, v in self.state_dict().items()},
            }

        @classmethod
        def from_dict(cls, data: dict) -> "SportsGPT":
            m = cls(
                vocab_size=data["vocab_size"],
                embed_dim=data["embed_dim"],
                num_heads=data["num_heads"],
                num_layers=data["num_layers"],
                max_len=data["max_len"],
            )
            state_dict = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
            m.load_state_dict(state_dict)
            return m

        def save(self, file_path: str | Path) -> None:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

        @classmethod
        def load(cls, file_path: str | Path) -> "SportsGPT":
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            return cls.from_dict(data)

else:
    class SportsGPT:
        """Full GPT model for sports commentary generation built from scratch."""

        def __init__(self, vocab_size: int = 300, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 1, max_len: int = 128, seed: int = 42):
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.max_len = max_len

            self.embeddings = SportsEmbeddings(vocab_size, embed_dim, max_len=max_len, seed=seed)
            self.blocks = [TransformerBlock(embed_dim, num_heads=num_heads, max_len=max_len, seed=seed + i) for i in range(num_layers)]
            self.ln_f = LayerNorm(embed_dim)

            rng = random.Random(seed + 99)
            std = 0.02
            self.w_head = [[rng.gauss(0, std) for _ in range(vocab_size)] for _ in range(embed_dim)]
            self.b_head = [0.0] * vocab_size

        def forward(self, idx: list | SimpleTensor, targets: list | SimpleTensor | None = None):
            import math
            raw_idx = idx.data if isinstance(idx, SimpleTensor) else idx
            if isinstance(raw_idx[0], int):
                raw_idx = [raw_idx]

            B = len(raw_idx)
            T = len(raw_idx[0])

            # 1. Embeddings
            x = self.embeddings(raw_idx)

            # 2. Transformer blocks
            for block in self.blocks:
                x = block(x)

            # 3. Final layer norm
            x = self.ln_f(x)
            raw_x = x.data if isinstance(x, SimpleTensor) else x

            # 4. LM Head projection
            all_logits = []
            for b in range(B):
                h_b = raw_x[b]  # (T, D)
                logits_b = matmul_2d(h_b, self.w_head)
                for t in range(T):
                    for v in range(self.vocab_size):
                        logits_b[t][v] += self.b_head[v]
                all_logits.append(logits_b)

            logits = SimpleTensor(all_logits)
            if targets is None:
                return logits

            # Cross-Entropy Loss computation
            raw_targets = targets.data if isinstance(targets, SimpleTensor) else targets
            if isinstance(raw_targets[0], int):
                raw_targets = [raw_targets]

            total_loss = 0.0
            total_tokens = 0
            for b in range(B):
                for t in range(T):
                    y = raw_targets[b][t]
                    probs = softmax_1d(all_logits[b][t])
                    p_target = max(probs[y], 1e-12)
                    total_loss += -math.log(p_target)
                    total_tokens += 1

            avg_loss = total_loss / max(total_tokens, 1)
            return logits, avg_loss

        def train_step(self, x_tokens: list, y_targets: list, lr: float = 0.08) -> float:
            """Execute a single backpropagation training step on the sports text."""
            if isinstance(x_tokens[0], int):
                x_tokens = [x_tokens]
            if isinstance(y_targets[0], int):
                y_targets = [y_targets]

            B = len(x_tokens)
            T = len(x_tokens[0])
            D = self.embed_dim
            V = self.vocab_size

            logits, loss = self.forward(x_tokens, targets=y_targets)
            raw_logits = logits.data

            x = self.embeddings(x_tokens)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            h = x.data

            grad_w = [[0.0] * V for _ in range(D)]
            grad_b = [0.0] * V
            norm_factor = 1.0 / (B * T)

            for b in range(B):
                for t in range(T):
                    probs = softmax_1d(raw_logits[b][t])
                    y = y_targets[b][t]
                    h_vec = h[b][t]

                    for v in range(V):
                        dz = (probs[v] - (1.0 if v == y else 0.0)) * norm_factor
                        grad_b[v] += dz
                        for d in range(D):
                            grad_w[d][v] += h_vec[d] * dz

            for d in range(D):
                for v in range(V):
                    self.w_head[d][v] -= lr * grad_w[d][v]
            for v in range(V):
                self.b_head[v] -= lr * grad_b[v]

            return loss

        def to_dict(self) -> dict:
            return {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_len": self.max_len,
                "w_head": self.w_head,
                "b_head": self.b_head,
            }

        @classmethod
        def from_dict(cls, data: dict) -> "SportsGPT":
            m = cls(
                vocab_size=data["vocab_size"],
                embed_dim=data["embed_dim"],
                num_heads=data["num_heads"],
                num_layers=data["num_layers"],
                max_len=data["max_len"],
            )
            m.w_head = data["w_head"]
            m.b_head = data["b_head"]
            return m

        def save(self, file_path: str | Path) -> None:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

        @classmethod
        def load(cls, file_path: str | Path) -> "SportsGPT":
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            return cls.from_dict(data)


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
