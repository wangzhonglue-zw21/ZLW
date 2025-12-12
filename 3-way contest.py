#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: reflection_to_new_prompt.ipynb
Conversion Date: 2025-12-12T23:30:12.924Z
"""

import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from datasets import load_dataset

# Silence HuggingFace warnings
hf_logging.set_verbosity_error()

# ==========================================
# 🤖 EXPERIMENT AGENT
# ==========================================
class BudgetComparisonAgent:
    def __init__(self, manager_model="Qwen/Qwen2.5-7B-Instruct", worker_model="Qwen/Qwen2.5-Math-7B-Instruct", device="cuda"):
        print(f"🚀 Loading Models on {device}...")
        self.device = device

        # MANAGER: The Analyst
        self.m_tokenizer = AutoTokenizer.from_pretrained(manager_model, trust_remote_code=True)
        self.m_model = AutoModelForCausalLM.from_pretrained(
            manager_model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        ).eval()

        # WORKER: The Solver
        self.w_tokenizer = AutoTokenizer.from_pretrained(worker_model, trust_remote_code=True)
        self.w_model = AutoModelForCausalLM.from_pretrained(
            worker_model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        ).eval()

        self.examples = []

        # Multivariate Regression Data
        self.X_train = []
        self.y_train = []
        self.weights = None
        self.is_fitted = False

    def _call_manager(self, prompt, max_new_tokens=1000):
        try:
            msgs = [{"role": "user", "content": prompt}]
            text_input = self.m_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.m_tokenizer(text_input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.m_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7)
            return self.m_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Manager Error: {e}")
            return ""

    def _call_worker(self, problem_text, budget=4096):
        try:
            prompt = f"Problem: {problem_text}\n\nLet's think step by step.\nAnswer in \\boxed{{}}."
            msgs = [{"role": "user", "content": prompt}]
            text_input = self.w_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.w_tokenizer(text_input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.w_model.generate(**inputs, max_new_tokens=budget, temperature=0.7)
            response = self.w_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            tokens_used = len(self.w_tokenizer.encode(response))
            return response, tokens_used
        except Exception as e:
            print(f"Worker Error: {e}")
            return "", 0

    def add_example(self, problem_text, actual_tokens):
        self.examples.append({"problem": problem_text[:200], "tokens": actual_tokens})

    def get_examples_text(self):
        if not self.examples: return "No examples yet."
        display_examples = self.examples[-5:]
        text = "=== REFERENCE ANCHORS ===\n"
        for i, ex in enumerate(display_examples, 1):
            text += f"{i}. Problem: {ex['problem']}... | Actual Cost: {ex['tokens']} tokens\n"
        return text

    def _extract_ratings(self, problem_text):
        """Extract ratings without making predictions"""
        examples = self.get_examples_text()
        prompt = f"""
You are a mathematical hardness analyst. Your task is to evaluate problem difficulty parameters to feed into a regression model.

[REFERENCE ANCHORS]
{examples}

[PROBLEM]
{problem_text[:400]}...

STEP 1: PARAMETER ASSESSMENT
Rate each parameter on 1-10 scale using these precise definitions:

1. **Conceptual Depth**: Prerequisite knowledge required (1=Basic, 10=Cutting Edge).
2. **Computational Effort**: Estimated number of steps (1=Simple, 10=Recursion/100+ steps).
3. **Abstraction Level**: Symbolic vs concrete (1=Real world, 10=Axiomatic).
4. **Heuristic Requirements**: Need for creativity (1=Standard, 10=Novel approach).
5. **Constrainedness**: Solution space narrowness (1=Open, 10=Unique path).

STEP 2: OUTPUT (EXACT FORMAT)
[CONCEPTUAL_DEPTH] X
[COMPUTATIONAL_EFFORT] X
[ABSTRACTION] X
[HEURISTIC] X
[CONSTRAINEDNESS] X

Reasoning: (Brief justification)
"""
        response = self._call_manager(prompt)

        # Parse Ratings
        ratings_vector = [1.0]  # Bias term
        keys = ["CONCEPTUAL_DEPTH", "COMPUTATIONAL_EFFORT", "ABSTRACTION", "HEURISTIC", "CONSTRAINEDNESS"]

        for k in keys:
            match = re.search(f'\[{k}\]\s*(\d+)', response)
            val = float(match.group(1)) if match else 5.0
            ratings_vector.append(val)

        return ratings_vector

    def train_regression(self):
        """Performs Multivariate Linear Regression with Ridge regularization"""
        if len(self.X_train) < 6:
            print("⚠️ Not enough data. Using reasonable defaults.")
            # Defaults based on intuition: base cost + moderate weights
            self.weights = np.array([50.0, 20.0, 40.0, 15.0, 25.0, 10.0])
            self.is_fitted = True
            return

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # Ridge regression with small regularization to prevent overfitting
        alpha = 10.0  # Regularization strength
        XtX = X.T @ X
        Xty = X.T @ y

        # Add regularization to diagonal (except bias term)
        reg_matrix = alpha * np.eye(XtX.shape[0])
        reg_matrix[0, 0] = 0  # Don't regularize bias

        # Solve: (X^T X + αI) w = X^T y
        self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        self.is_fitted = True

        print(f"📊 Regression Trained! Learned Weights:")
        params = ["Bias", "Depth", "Effort", "Abs", "Heur", "Const"]
        for p, w in zip(params, self.weights):
            print(f"   {p}: {w:.2f}")

    def predict_rigid(self, problem_text):
        """Make predictions using trained regression"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet! Call train_regression() first.")

        ratings_vector = self._extract_ratings(problem_text)
        budget = int(np.dot(np.array(ratings_vector), self.weights))
        budget = max(budget, 50)  # Minimum budget

        return budget, ratings_vector

    # ==========================================
    # 🧠 PROMPT 2: REFINED INTUITIVE
    # ==========================================
    def predict_intuitive(self, problem_text):
        examples = self.get_examples_text()
        prompt = f"""
You are a mathematical hardness analyst. Evaluate difficulty and predict resource usage without solving.

[REFERENCE ANCHORS]
{examples}

Problem: {problem_text[:500]}...

Step 1: Identify problem type and core principles.

Step 2: **Decomposition**: Verbally list the distinct logical steps or sub-questions the solver MUST answer.

Step 3: Rate standard parameters (1-10):
- Conceptual Depth
- Computational Effort (Base this on Step 2 list size)
- Abstraction Level
- Heuristic Requirements
- Constrainedness

Step 4: **X-Factor**: Detect unique features (e.g., repetitive verification, pattern enumeration) that might explode token usage.

Step 5: Synthesize ratings, X-Factors, and Anchor comparisons to output budget.

Format your answer exactly as:
[CONCEPTUAL_DEPTH] X
[COMPUTATIONAL_EFFORT] X
[ABSTRACTION] X
[HEURISTIC] X
[CONSTRAINEDNESS] X
[TOKENS] X
"""
        response = self._call_manager(prompt)
        match = re.search(r'\[TOKENS\]\s*(\d+)', response)
        budget = int(match.group(1)) if match else 0
        return budget, response

    # ==========================================
    # 🧪 PROMPT 3: META-ARCHITECT
    # ==========================================
    def predict_meta_architect(self, problem_text):
        examples = self.get_examples_text()
        prompt = f"""
You are a 'Metacognitive Risk Analyst'. Predict the **safe token budget**, accounting for potential brute-force traps.

[REFERENCE ANCHORS]
{examples}

[PROBLEM]
{problem_text[:600]}...

[ANALYSIS PROTOCOL]

### STEP 1: DUAL-PATHWAY SIMULATION (Tree of Thoughts)
Simulate two distinct ways a solver might attack this.
* **Path A (The "Olympiad" Approach):** The elegant trick or formula.
* **Path B (The "Grind" Approach):** Step-by-step calculation/iteration if the insight is missed.
* *Critical Decision:* Which path is a standard LLM *most likely* to fall into?

### STEP 2: STRUCTURAL & VOLUME DIAGNOSIS
Analyze the *physical volume* of the **Likely Path**.
* **Decomposition:** List the logical "Work Units" (e.g., "1. Expand Eq, 2. Sub-case 1...").
* **Trap Detection:** Does this involve [Recursion], [Pattern Enumeration], or [Telescoping Series]?
    * *Heuristic:* If "Pattern Enumeration" is present, the token cost is often 3x the difficulty rating.

### STEP 3: FINAL PREDICTION
Output the budget.
* **Buffer Rule:** If Path A and Path B diverge significantly (Trick vs Brute Force), budget for **Path B**.

[OUTPUT FORMAT]
[PATH_ANALYSIS] <Brief comparison>
[LIKELY_PATH] <"Olympiad" or "Grind">
[TRAP_DETECTED] <Type or None>
[CONCEPTUAL_DEPTH] X
[COMPUTATIONAL_EFFORT] X
[ABSTRACTION] X
[HEURISTIC] X
[CONSTRAINEDNESS] X
[TOKENS] <Integer>
"""
        response = self._call_manager(prompt)
        match = re.search(r'\[TOKENS\]\s*(\d+)', response)
        budget = int(match.group(1)) if match else 0

        trap = re.search(r'\[TRAP_DETECTED\]\s*(.+)', response)
        trap_val = trap.group(1).strip() if trap else "None"
        path = re.search(r'\[LIKELY_PATH\]\s*(.+)', response)
        path_val = path.group(1).strip() if path else "?"

        return budget, trap_val, path_val, response

# ==========================================
# 📊 COMPARISON RUNNER
# ==========================================
def run_ab_test():
    print(f"\n{'='*70}")
    print("🧪 3-WAY SHOWDOWN: MULTIVARIATE REGRESSION vs INTUITIVE vs META-ARCHITECT")
    print(f"{'='*70}\n")

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    iterator = iter(ds.skip(4000))

    agent = BudgetComparisonAgent()

    # --- Phase 1: Bootstrap & Train (10 Problems) ---
    print("📚 Bootstrapping & Training Regression (10 problems)...")
    for i in range(10):
        row = next(iterator)
        print(f"   Solving Anchor {i+1}...", end="\r")
        _, actual = agent._call_worker(row['problem'])

        # Extract ratings and store for training
        ratings = agent._extract_ratings(row['problem'])
        agent.X_train.append(ratings)
        agent.y_train.append(actual)

        agent.add_example(row['problem'], actual)
        print(f"   ✅ Anchor {i+1} Cost: {actual} tokens      ")

    # NOW train the regression with collected data
    agent.train_regression()

    # --- Phase 2: Head-to-Head (20 Problems) ---
    print(f"\n🥊 STARTING COMPARISON (20 Problems)\n")

    errors_rigid = []
    errors_intuit = []
    errors_meta = []

    for i in range(20):
        row = next(iterator)
        problem = row['problem']

        print(f"--- Problem {i+1} ---")
        print(f"Excerpt: {problem[:60]}...")

        # 1. Ground Truth
        print("   Thinking...", end="\r")
        _, actual = agent._call_worker(problem)
        print(f"   🎯 ACTUAL COST:     {actual} tokens")

        # 2. Rigid Formula (Learned Weights)
        budget_r, _ = agent.predict_rigid(problem)
        diff_r = abs(budget_r - actual)
        errors_rigid.append(diff_r)

        # 3. Refined Intuitive
        budget_i, _ = agent.predict_intuitive(problem)
        diff_i = abs(budget_i - actual)
        errors_intuit.append(diff_i)

        # 4. Meta-Architect
        budget_m, trap_m, path_m, _ = agent.predict_meta_architect(problem)
        diff_m = abs(budget_m - actual)
        errors_meta.append(diff_m)

        # Round Summary
        print(f"   🔵 Rigid (Reg):      {budget_r} (Err: {diff_r})")
        print(f"   🟠 Intuitive:        {budget_i} (Err: {diff_i})")
        print(f"   🟣 Meta-Arch:        {budget_m} (Err: {diff_m}) | Path: {path_m}")

        min_err = min(diff_r, diff_i, diff_m)
        if min_err == diff_m: print("   🏆 Winner: META-ARCHITECT")
        elif min_err == diff_i: print("   🏆 Winner: INTUITIVE")
        else: print("   🏆 Winner: RIGID")
        print("-" * 40)

    # --- Final Stats ---
    avg_r = sum(errors_rigid) / len(errors_rigid)
    avg_i = sum(errors_intuit) / len(errors_intuit)
    avg_m = sum(errors_meta) / len(errors_meta)

    print(f"\n{'='*70}")
    print("🏁 FINAL SCOREBOARD (Average Absolute Error)")
    print(f"🔵 Rigid Formula (MLR): {avg_r:.1f}")
    print(f"🟠 Refined Intuit:      {avg_i:.1f}")
    print(f"🟣 Meta-Architect:      {avg_m:.1f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_ab_test()
    else:
        print("Comparison requires CUDA/GPU.")

import types
import re

# 1. Define the new prediction logic (The "Challenger")
def predict_meta_synthesis(self, problem_text):
    examples = self.get_examples_text()

    # Your exact requested prompt
    prompt = f"""
You are a mathematical hardness analyst and expert resource predictor. Your role is to evaluate difficulty and predict the required token budget by performing a comprehensive, **meta-reasoning synthesis.**

[REFERENCE ANCHORS]
{examples}

Problem: {problem_text[:400]}...

Step 1: **Categorization and Structural Analysis**
**First, classify the problem (e.g., Algebra, Geometry, Combinatorics, Number Theory).** Next, identify the core principles. **Identify any recurring mathematical forms or subtle patterns that suggest either a short solution using identity/induction, or a long solution requiring deep recursive steps.**

Step 2: Break down the problem into its main components verbally. List the set of subproblems the solver has to go through.

Step 3: Rate each parameter (1-10 scale):
- Conceptual Depth: Prerequisite knowledge level
- Computational Effort: Calculation complexity
- Abstraction Level: Real-world detachment
- Heuristic Requirements: Creative strategies needed
- Constrainedness: Solution space narrowness

Step 4: **Complexity Assessment and Special Characteristics**
Identify any unique features that affect token usage but are not captured by standard parameters. **Critically assess the forward-looking complexity: Does the problem structure suggest a lengthy, iterative validation process (like many case analyses or recursive steps), which would significantly increase the final token count? Is the problem in a certain form that the LRM today has to take more steps to formally reason?**

Step 5: Compare your analysis to the Reference Anchors.
- Is this problem easier or harder than any similar problem in anchors? By how much? **When comparing, focus on *why* a successful anchor required its cost (long setup, complex intermediate steps, detailed final proof) to inform your current prediction.**

Step 6: **Final Synthesis and Budget Output**
Based on this comprehensive analysis, including your parameter ratings, the problem category, the complexity assessment, and the anchor comparisons, output the final token budget.
(Use the examples as calibration points, not fixed rules)

Format your answer exactly as:
[CONCEPTUAL_DEPTH] X
[COMPUTATIONAL_EFFORT] X
[ABSTRACTION] X
[HEURISTIC] X
[CONSTRAINEDNESS] X
[TOKENS] X
"""
    response = self._call_manager(prompt)

    # Robust Parsing: Handles ":", spaces, and non-digit characters around the number
    match = re.search(r'\[TOKENS\]\s*:?\s*(\d+)', response)
    budget = int(match.group(1)) if match else 0

    return budget, response

# 2. Monkey-patch it into your existing agent instance
if 'agent' in globals():
    agent.predict_meta_synthesis = types.MethodType(predict_meta_synthesis, agent)
    print("✅ Successfully attached 'predict_meta_synthesis' to existing agent.")
else:
    print("⚠️ 'agent' object not found. Please run the initialization cell first.")

# 3. Run the "Meta-Synthesis" Comparison Loop
if 'agent' in globals():
    print(f"\n{'='*70}")
    print("🧪 TEST RUN: META-REASONING SYNTHESIS PROMPT")
    print(f"{'='*70}\n")

    # Use a fresh slice of the dataset to avoid repeats (Skip 4050)
    try:
        fresh_iterator = iter(ds.skip(4050))
    except NameError:
        print("Dataset 'ds' not found. Please reload the dataset.")
        fresh_iterator = []

    results_meta = []

    for i in range(5):
        try:
            row = next(fresh_iterator)
        except StopIteration:
            break

        problem = row['problem']
        print(f"--- Problem {i+1} ---")
        print(f"Excerpt: {problem[:60]}...")

        # 1. Ground Truth (Worker)
        print("   Thinking...", end="\r")
        _, actual = agent._call_worker(problem)
        print(f"   🎯 ACTUAL COST:     {actual} tokens")

        # 2. Your New Prompt (Meta-Synthesis)
        budget, response = agent.predict_meta_synthesis(problem)
        diff = abs(budget - actual)
        results_meta.append(diff)

        # Extract Category snippet for display
        lines = response.split('\n')
        category_snippet = next((line for line in lines if "Algebra" in line or "Geometry" in line or "Number" in line), "Unknown Category")[:60]

        print(f"   🟣 Meta-Synthesis:  {budget} (Err: {diff})")
        print(f"   📝 Classification:  {category_snippet}...")
        print("-" * 40)

    avg_err = sum(results_meta) / len(results_meta) if results_meta else 0
    print(f"\n🏁 Average Error for Meta-Synthesis: {avg_err:.1f}")