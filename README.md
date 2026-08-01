# micro-grad

A deep learning framework built from scratch in Python/NumPy (with optional CUDA support via CuPy), created as a personal project to learn how AI systems actually work under the hood — from automatic differentiation up through transformer-based language models.

## What's inside

- **mtorch/** — the core framework: tensors with autograd, neural network layers, optimizers, and training utilities.
- **experiments/** — small, self-contained scripts that use the framework to train real models, including a classifier, a digit recognizer, a sequence-to-sequence translator, and a character-level chatbot.

## Why this exists

This project was a hands-on way to learn the fundamentals of AI programming by implementing them directly, rather than only using existing libraries.

## Requirements

See `requirements.txt`. GPU acceleration is optional and falls back to CPU automatically if unavailable.
