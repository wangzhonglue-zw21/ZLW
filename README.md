# Verbal Reinforcement Learning Experiments

This repository contains experimental implementations exploring verbal reinforcement learning and self-improving AI systems for mathematical problem difficulty prediction.

## Files

### `3_way_contest.py`
Compares performance of rigid mathematical regression estimation vs. intuitive verbal reasoning estimation for predicting problem difficulty.

### `Batch_prompt_improvement_Large.py`
Implements a self-improving AI system where a "Manager" LLM learns to predict math problem difficulty by analyzing its prediction errors, generating an optimized prediction prompt through reflection, and validating the improvement on new problems - achieving a 37.5% reduction in prediction error.

### `Prompt_evolution_miniature_feature_matching.py`
Demonstrates the ineffectiveness of verbal learning through matching miniature/surface features (keyword matching) - an approach similar to methods used in standard deep learning but shown to be inadequate for verbal reasoning tasks.

### `prompt_evolution_high_abstraction.py`
Used for comparison against `Prompt_evolution_miniature_feature_matching.py`, demonstrating that high-level abstractions and summaries work better as parameters for verbal reinforcement learning. Shows that a certain level of interpretation is vital to the effectiveness of learning.

### `prototype_token_predictor_with_confidence.py`
A prototype that incorporates verbal learning with standard mathematical deep learning tools for a self-improving token estimator.

## Key Findings

- Verbal reasoning can achieve significant improvements (37.5% error reduction) through reflection-based prompt optimization
- Surface-level feature matching (keywords) is ineffective for verbal learning
- High-level abstractions and interpretation are crucial for effective verbal reinforcement learning
- Hybrid approaches combining verbal reasoning with traditional ML tools show promise
