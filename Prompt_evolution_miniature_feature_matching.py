#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: 3-way contest.ipynb
Conversion Date: 2025-12-12T23:49:23.744Z
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class SelfReflectiveOptimizer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # INITIAL PROMPT (The "Bad" one to start with)
        self.current_prompt = """You are a mathematical hardness analyst. Your goal is to predict the token budget for a solver model.

[CONTEXT INFO]
Problems in this specific problem set typically require between **200 and 2000 tokens** to solve.

[PROBLEM]
{problem_text}

Step 1: **Categorization and Structural Analysis**
First, classify the problem. Identify recurring mathematical forms.

Step 2: **Mental Simulation**
Break down the problem into its main components verbally.

Step 3: **Complexity Assessment**
Rate each parameter (1-10 scale):
- Conceptual Depth
- Computational Effort
- Abstraction Level
- Heuristic Requirements
- Constrainedness

Step 4: **Final Synthesis and Budget Output**
Based on this comprehensive analysis, output the final token budget.

Output your entire thought process followed by:
[TOKENS] <Integer>"""

    def _generate(self, prompt, max_tokens=1500, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.05
            )
        # Decode only the new tokens
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    # --- PHASE 1: PREDICT (Manager) ---
    def predict_tokens(self, problem):
        print("\n🔵 MANAGER IS PREDICTING...")

        # Insert problem into current system prompt
        if "{problem_text}" in self.current_prompt:
            prompt_text = self.current_prompt.replace("{problem_text}", problem)
        else:
            prompt_text = self.current_prompt + f"\n\n[PROBLEM]\n{problem}"

        response = self._generate(prompt_text)

        # Extract the integer prediction
        match = re.search(r"\[TOKENS\]\s*(\d+)", response)
        predicted_val = int(match.group(1)) if match else 0

        # Clean up response for storage (stop at the token tag)
        clean_reasoning = response
        if match:
            clean_reasoning = response[:match.end()]

        return predicted_val, clean_reasoning

    # --- PHASE 2: SOLVE (Worker) ---
    def worker_solve(self, problem):
        print("🤖 WORKER IS SOLVING...")

        # Standard solver prompt
        worker_prompt = (
            "You are a math solver. Solve the following problem step-by-step.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )

        solution = self._generate(worker_prompt, max_tokens=2048)

        # Count ACTUAL tokens used (approximate via tokenizer)
        actual_tokens = len(self.tokenizer.encode(solution))

        return actual_tokens, solution

    # --- PHASE 3: REFLECT (Prompt Engineer) ---
    def optimize_prompt(self, problem, manager_reasoning, predicted, worker_solution, actual):
        print("\n🧬 REFLECTION & OPTIMIZATION TRIGGERED...")

        # 1. Construct the Experiment Result History
        experiment_result = f"""
[EXPERIMENT DATA]
PROBLEM:
{problem}

CURRENT MANAGER PROMPT:
{self.current_prompt}

MANAGER REASONING (FAILED):
{manager_reasoning}

MANAGER PREDICTION: {predicted} tokens

WORKER ACTUAL SOLUTION:
{worker_solution[:1000]}... (truncated)

WORKER ACTUAL COST: {actual} tokens
"""

        # 2. The "Meta-Instruction" (The fixed method from our chat)
        user_instruction = f"""
Consider the experiment data above.
The "Manager" (Predictor) failed significantly. It predicted {predicted}, but actual was {actual}.
The current prompt focuses on abstract ratings (1-10) which are not correlated with text length.

**Your Task:**
1. **Analyze:** Why did the prompt fail?
2. **Improve:** Write a NEW System Prompt.
   - Remove abstract 1-10 scales.
   - Force the model to scan for keywords and assign specific token costs.
   - Ensure it outputs a step-by-step summation.

[CRITICAL CONSTRAINTS]
- STOP immediately after the prompt
- Keep {{problem_text}} placeholder
- Output only: ### ANALYSIS and ### NEW SYSTEM PROMPT, the prompt should be design to predict token usage to solve any math problem, not solve the problem.

[OUTPUT FORMAT]
### ANALYSIS
(Single sentence)
### NEW SYSTEM PROMPT
(Full prompt text)
"""

        # 3. Ask Qwen to fix it
        messages = [
            {"role": "system", "content": "You are an Expert Prompt Engineer and LLM Optimizer."},
            {"role": "user", "content": experiment_result + "\n\n" + user_instruction}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate the optimization
        optimization_response = self._generate(text_input, max_tokens=1500)

        return optimization_response

    # --- MAIN CYCLE ---
    def run_optimization_loop(self, problem_text):
        # 1. Predict
        pred_val, pred_text = self.predict_tokens(problem_text)
        print(f"   -> Predicted: {pred_val}")

        # 2. Solve (Ground Truth)
        actual_val, solution_text = self.worker_solve(problem_text)
        print(f"   -> Actual: {actual_val}")

        # 3. Check Deviation
        error = abs(pred_val - actual_val)
        print(f"   -> Deviation: {error}")

        # 4. Reflect if error is high (e.g., > 300 tokens)
        if error > 300:
            print("   -> ⚠️ LARGE ERROR DETECTED. OPTIMIZING PROMPT...")

            optimization_output = self.optimize_prompt(
                problem=problem_text,
                manager_reasoning=pred_text,
                predicted=pred_val,
                worker_solution=solution_text,
                actual=actual_val
            )

            print("="*40)
            print("NEW OPTIMIZED PROMPT CANDIDATE:")
            print(optimization_output)
            print("="*40)

            # Simple extraction to update the prompt for next time
            if "### NEW SYSTEM PROMPT" in optimization_output:
                new_prompt = optimization_output.split("### NEW SYSTEM PROMPT")[1].strip()
                self.current_prompt = new_prompt
                print("✅ PROMPT UPDATED IN MEMORY.")
        else:
            print("✅ Prediction was acceptable.")

# --- EXECUTION ---
if __name__ == "__main__":
    optimizer = SelfReflectiveOptimizer()

    # The Test Problem
    math_problem = r"""
Given a sequence $\{a_n\}$ with each term being a positive number, and satisfying $a_2=5$, $$a_{n+1}=a_{n}^2-2na_n+2 \quad (n \in \mathbb{N}^*).$$
(1) Conjecture the general formula for $\{a_n\}$.
(2) Let $b_n=2^{n-1}$ and $c_n=a_n+b_n$, find the sum of the first $n$ terms of the sequence $\{c_n\}$, denoted as $T_n$.
"""

    optimizer.run_optimization_loop(math_problem)