# To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models

---

## 🔍 Overview

Reinforcement Learning with Verifiable Rewards (RLVR) has proven highly effective for inducing expert-level reasoning in specific domains such as mathematics and code generation. However, **how to best combine RLVR across multiple domains** remains an open question.

This project presents detailed comparison and analysis about these paradigms for multi-domain RLVR:
1. **Mixed multi-task RLVR**  
2. **Separate domain-specific RLVR followed by model merging**

We choose multiple commonly used high-level tasks (e.g., **math, coding, science, and instruction following**) as our target domains, and provide in-depth analysis of:
- Training efficiency
- Weight space geometry
- Model prediction behavior
- Information constraints

---

## ✨ Key Findings

- Mixed multi-task RLVR achieves comparable performance with separate RLVR followed by model merging, while using only **33.2% of GPU hours**.
- Cross-domain RLVR shows **minimal interference**; reasoning-intensive domains (math, coding, science) exhibit **strong synergistic effects**.
- Weight update footprints across domains **significantly overlap**, with positive cosine similarity after random projection.
- Performance is better explained by **policy neighborhood transfer** rather than KL divergence magnitude alone.
- Model merging primarily inherits the original capabilities of the single-task models, whereas multi-task training exhibits a larger divergence to the single-task ones.

---

## 🧪 Experimental Setup

### Base Model
- **Qwen3-4B-Base**

### Domains
- Math
- Coding
- Science
- Instruction Following

### Training Pipeline
1. Supervised Fine-Tuning (SFT) on ~14M samples  
2. Reinforcement Learning with Verifiable Rewards (RLVR) using **GRPO**

### Datasets
- Nemotron-3 Nano SFT datasets: https://huggingface.co/collections/nvidia/nemotron-post-training-v3
- Nemotron-3 Nano RL datasets: https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend

---

## 🔁 Compared Paradigms

### Multi-Task RLVR
- Mixed-domain reward optimization
- Single unified policy

### Model Merging
- Average merging
- Task arithmetic
- TIES / TIES + DARE
- SCE
- Multi-teacher on-policy distillation (MT-OPD)

---

## 📊 Evaluation Benchmarks

- **Math**: AIME’24, AIME’25  
- **Coding**: LiveCodeBench v5 / v6  
- **Science**: HLE, GPQA-Diamond  
- **Instruction Following**: IFEval, IFBench  
- **General**: MMLU-Redux  

Metrics: Avg@K accuracy

---

## 🔬 Analysis Modules

This repository includes analysis code for:

- Weight shift overlap & cosine similarity
- Policy KL cross-comparison
- Policy neighborhood identification
- Skill overlap between multi-task and single-task models
- Outcome-based vs process-based verification
- Emergent self-discrimination behavior

---

## 📁 Repository Structure

```text
.
├── configs/
└── README.md
