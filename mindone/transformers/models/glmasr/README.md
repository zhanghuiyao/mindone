# GLM-ASR Model for MindSpore

This directory contains the MindSpore implementation of the GLM-ASR (Audio Speech Recognition) model, adapted from the HuggingFace Transformers library.

## Files

- configuration_glmasr.py: Configuration classes for GLM-ASR model and encoder
- modeling_glmasr.py: MindSpore implementation of GLM-ASR model
- processing_glmasr.py: Processor for handling audio and text inputs
- modular_glmasr.py: Modular version (reference, copied from transformers)

## Key Changes from PyTorch Version

1. Import Changes: torch to mindspore, torch.nn to mindspore.mint.nn, nn.Module to mindspore.nn.Cell
2. Method Changes: forward() to construct()
3. Tensor Operations: transpose() to swapaxes() where appropriate
4. Removed PyTorch-specific decorators and features
5. Processor updated to use MindSpore tensors

## Model Architecture

GLM-ASR combines an audio encoder, multi-modal projector, and language model for audio-to-text tasks.

## Testing

Run basic structure tests: python tests/transformers_tests/models/glmasr/test_basic.py

## References

- Original: https://github.com/huggingface/transformers
- Model: https://huggingface.co/zai-org/GLM-ASR-Nano-2512
