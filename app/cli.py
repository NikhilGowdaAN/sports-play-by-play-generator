"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from app.core import milestone_1, get_corpus_stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    corpus_parser = sub.add_parser("corpus", help="view sports commentary corpus details and preview")
    corpus_parser.add_argument("--preview", action="store_true", help="show sample commentary preview")

    # TODO: add your own commands here as you work through the milestones.
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


if __name__ == "__main__":
    main()
