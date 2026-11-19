# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/qfpysaehaz/badge.svg)](https://ask.thili.ai/b/qfpysaehaz)

This project guides you through building a GPT model from scratch to create a command-line assistant for sports fans. You'll train it on real sports news to generate dynamic commentary snippets for fictional game scenarios, applying core concepts like tokenization, embeddings, and causal self-attention.

## What you are building

**Goal.** Build a command-line assistant that generates short, dynamic sports commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** The project is complete when a command-line interface allows users to input a prompt and receive generated, short, dynamic sports commentary snippets, demonstrating a fully trained GPT model built from scratch.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Raw text input preparation.
- [x] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Byte-pair tokenizer and embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal self-attention and transformer block assembly.
- [ ] **4. Train the GPT model using the sports text data.**  
      Hint: Model training.
- [ ] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Text generation.
- [ ] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: Applying text generation in a CLI.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify a desired length for the generated commentary.
- Implement a loop in the CLI for continuous generation without restarting.
- Experiment with different hyperparameters during training to observe changes in commentary style.
