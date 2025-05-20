# Project 2: Aligning with RLHF and DPO

This project focuses on aligning a pretrained Large Language Model (LLM) with human preferences using Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

## Objectives
- Align an LLM to follow human instructions.
- Compare RLHF and DPO in terms of performance.

## Steps
1. **Prepare a Preference Dataset**:
   - Use a dataset with pairwise preferences (e.g., "Response A is better than Response B").
   - Example: Synthetic data or a subset of Alpaca.
2. **Implement RLHF**:
   - Train a reward model on the preference dataset.
   - Use Proximal Policy Optimization (PPO) to fine-tune the LLM.
   - **Code Placeholder**: Implement RLHF in `rlhf.py`.
3. **Implement DPO**:
   - Optimize the LLM directly on the preference dataset.
   - **Code Placeholder**: Implement DPO in `dpo.py`.
4. **Evaluate**:
   - Test the aligned models on a dialogue task (e.g., answering prompts).
   - Compare performance using human evaluation or automated metrics.

## Dataset
- Alpaca (`huggingface/datasets/alpaca`).
- UltraFeedback (`huggingface/datasets/ultrafeedback`).

## Run
```bash
python src/section_2_training/dpo.py
```

## Expected Output
- Two aligned models (RLHF and DPO).
- Comparison of their performance on a dialogue task.

## Next Steps
Proceed to [Section 3: Scaling Model Training](section_3_scaling.md).
