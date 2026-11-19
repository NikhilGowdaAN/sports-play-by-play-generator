"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    get_corpus_stats,
    BPETokenizer,
    SportsEmbeddings,
    TransformerBlock,
    SportsGPT,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    corpus_parser = sub.add_parser("corpus", help="view sports commentary corpus details and preview")
    corpus_parser.add_argument("--preview", action="store_true", help="show sample commentary preview")

    tokenize_parser = sub.add_parser("tokenize", help="tokenize sports text using the BPE tokenizer")
    tokenize_parser.add_argument("text", help="sports commentary text to tokenize")
    tokenize_parser.add_argument("--vocab-size", type=int, default=320, help="target vocabulary size")

    embed_parser = sub.add_parser("embed", help="generate token and positional embeddings for sports text")
    embed_parser.add_argument("text", help="sports commentary text to embed")
    embed_parser.add_argument("--embed-dim", type=int, default=64, help="embedding dimension")

    transformer_parser = sub.add_parser("transformer", help="pass sports text through causal attention and transformer block")
    transformer_parser.add_argument("text", help="sports commentary text to process")
    transformer_parser.add_argument("--embed-dim", type=int, default=64, help="embedding dimension")
    transformer_parser.add_argument("--heads", type=int, default=4, help="number of attention heads")

    train_parser = sub.add_parser("train", help="train the sports GPT model on commentary text")
    train_parser.add_argument("--steps", type=int, default=15, help="number of training optimization steps")
    train_parser.add_argument("--lr", type=float, default=0.08, help="learning rate")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")
    elif args.command == "corpus":
        corpus = milestone_1()
        stats = get_corpus_stats(corpus)
        print("Sports Commentary Corpus Statistics:")
        print(f"  Total Characters: {stats['char_count']}")
        print(f"  Total Words:      {stats['word_count']}")
        print(f"  Unique Words:     {stats['unique_word_count']}")
        print(f"  Non-empty Lines:  {stats['line_count']}")
        print(f"  Vocabulary Size:  {stats['vocab_size']}")
        if args.preview:
            print("\nPreview Sample:")
            print("-" * 40)
            sample = corpus[:350]
            print(f"{sample}...")
    elif args.command == "tokenize":
        corpus = milestone_1()
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=args.vocab_size)
        tokens = tokenizer.encode(args.text)
        decoded = tokenizer.decode(tokens)
        print(f"Input text:     '{args.text}'")
        print(f"UTF-8 Bytes:    {len(args.text.encode('utf-8'))}")
        print(f"Token count:    {len(tokens)}")
        print(f"Token IDs:      {tokens}")
        print(f"Decoded match:  {decoded == args.text}")
    elif args.command == "embed":
        m2 = milestone_2(embed_dim=args.embed_dim)
        tokenizer = m2["tokenizer"]
        embeddings = m2["embeddings"]
        tokens = tokenizer.encode(args.text)
        emb = embeddings(tokens)
        print(f"Input text:       '{args.text}'")
        print(f"Token count:      {len(tokens)}")
        print(f"Embedding shape:  {emb.shape}")
        print(f"Embedding vector dimension: {args.embed_dim}")
    elif args.command == "transformer":
        m2 = milestone_2(embed_dim=args.embed_dim)
        tokenizer = m2["tokenizer"]
        embeddings = m2["embeddings"]
        tokens = tokenizer.encode(args.text)
        emb = embeddings(tokens)
        m3 = milestone_3(embeddings_input=emb, embed_dim=args.embed_dim, num_heads=args.heads)
        print("Transformer Block Forward Pass:")
        print(f"  Input text:        '{args.text}'")
        print(f"  Token count:       {len(tokens)}")
        print(f"  Embedding shape:   {m3['input_shape']}")
        print(f"  Output shape:      {m3['output_shape']}")
        print(f"  Attention heads:   {m3['num_heads']}")
        print(f"  Causal masking:    {'PASSED (strict causality)' if m3['causal_mask_verified'] else 'FAILED'}")
        print(f"  Residual check:    {'PASSED (dimension preserved)' if m3['residual_verified'] else 'FAILED'}")
    elif args.command == "train":
        print(f"Training Sports GPT model for {args.steps} steps (lr={args.lr})...")
        m4 = milestone_4(steps=args.steps, lr=args.lr)
        print("Model Training on Sports Corpus:")
        print(f"  Vocabulary Size:   {m4['vocab_size']}")
        print(f"  Embedding Dim:     {m4['embed_dim']}")
        print(f"  Steps Trained:     {m4['trained_steps']}")
        print(f"  Initial Loss:      {m4['initial_loss']:.4f}")
        print(f"  Final Loss:        {m4['final_loss']:.4f}")
        print(f"  Loss Reduction:    {m4['loss_reduction']:.4f}")
        print(f"  Training Status:   {'PASSED (loss decreased)' if m4['training_verified'] else 'FAILED'}")


if __name__ == "__main__":
    main()
