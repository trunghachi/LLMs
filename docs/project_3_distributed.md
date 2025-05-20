# Project 3: Distributed Training with ZeRO

This project guides you through scaling the training of a Transformer model across multiple GPUs using PyTorch’s DistributedDataParallel (DDP) and ZeRO (Zero Redundancy Optimizer).

## Objectives
- Set up distributed training for an LLM.
- Use ZeRO to optimize memory usage.

## Steps
1. **Set Up a Multi-GPU Environment**:
   - Ensure you have access to multiple GPUs (e.g., via a cloud provider).
2. **Implement DDP with ZeRO**:
   - Use PyTorch’s DDP and the Accelerate library for ZeRO.
   - **Code Placeholder**: Implement distributed training in `distributed.py`.
3. **Train a Small Transformer**:
   - Use a subset of Wikitext.
   - Train for a few epochs and monitor loss across GPUs.
   - **Code Placeholder**: Add training loop in `distributed.py`.
4. **Evaluate**:
   - Check for consistent performance across GPUs.
   - Measure training speedup.

## Expected Output
- A trained Transformer model using distributed training.
- Speedup metrics compared to single-GPU training.

## Dataset
Wikitext (`huggingface/datasets/wikitext`).

## Run
```bash
python src/section_3_scaling/distributed.py
```
## Next Steps
Proceed to [Section 4: Fine-Tuning LLMs](section_4_finetuning.md).
