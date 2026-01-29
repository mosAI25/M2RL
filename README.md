# To Mix or To Merge: Multi-Domain Reinforcement Learning for Large Language Models

This repository contains the official implementation and analysis code for the paper:

**To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models**  


---

## 🔍 Overview

Reinforcement Learning with Verifiable Rewards (RLVR) has proven highly effective for inducing expert-level reasoning in specific domains such as mathematics and code generation.  
However, **how to best combine RLVR across multiple domains** remains an open question.

This work presents a **systematic comparison** between two dominant paradigms for multi-domain RLVR:

1. **Mixed multi-task RLVR**  
2. **Separate domain-specific RLVR followed by model merging or distillation**

We conduct extensive experiments across **math, coding, science, and instruction following**, and provide in-depth analysis of:
- Performance trade-offs
- Training efficiency
- Weight-space geometry
- Policy neighborhood interactions
- Emergent verification and self-discrimination abilities

---

## ✨ Key Findings

- **Mixed multi-task RLVR achieves comparable or better performance** than model merging, while using only **~33% of GPU hours**.
- Cross-domain RLVR shows **minimal interference**; reasoning-intensive domains (math, code, science) exhibit **strong synergistic effects**.
- Weight update footprints across domains **significantly overlap**, with positive cosine similarity after random projection.
- Performance is better explained by **policy neighborhood transfer** rather than KL divergence magnitude alone.
- Multi-task RLVR induces **emergent cross-domain verification synergy**, combining outcome-level intuition and process-level reasoning.

---

## 🧪 Experimental Setup

### Base Model
- **Qwen3-4B-Base**

### Domains
- Mathematics
- Coding
- Science
- Instruction Following

### Training Pipeline
1. Supervised Fine-Tuning (SFT) on ~14M samples  
2. Reinforcement Learning with Verifiable Rewards (RLVR) using **GRPO**

### Datasets
- Nemotron-3 Nano SFT & RLVR datasets
- DAPO, CodeContests, OpenScienceReasoning-2, WildChat-1M (see paper for full details)

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
