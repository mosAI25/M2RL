# To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models

---

## 🔍 Overview

Reinforcement Learning with Verifiable Rewards (RLVR) has proven highly effective for inducing expert-level reasoning in specific domains such as mathematics and code generation. However, **how to best combine RLVR across multiple domains** remains an open question.

This project presents detailed comparison and analysis about these paradigms for multi-domain RLVR:
1. **Mixed multi-task RLVR**  
2. **Separate domain-specific RLVR followed by model merging (weight merging or multi-teacher on-policy distillation)**

We choose multiple commonly used high-level tasks (e.g., **math, coding, science, and instruction following**) as our target domains, and provide in-depth analysis of:
- Training efficiency
- Weight space geometry
- Model prediction behavior
- Information constraints

Paper:

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

| SFT data blend | #Samples | Proportion (%) | Source Datasets | Sampling method |
| :--- | :--- | :--- | :--- | :--- |
| Formal Proofs | 335,122 | 2.37 | Nemotron-Math-Proofs-v1 | random sampling |
| Math | 1,878,601 | 13.30 | Nemotron-Math-v2 | random sampling |
| Math w/ Tools | 1,071,924 | 7.59 | Nemotron-Math-v2 | use all |
| Science | 2,263,340 | 16.04 | Nemotron-Science-v1 | repeat 10 times |
| Code | 3,927,984 | 27.81 | Nemotron-Competitive-Programming-v1 | use all |
| Chat | 4,309,780 | 30.52 | Nemotron-Instruction-Following-Chat-v1 | repeat 10 times |
| Conversational Agent | 335,122 | 2.37 | Nemotron-Agentic-v1 | use all |
| **Total** | **14,121,873** | **100.00** | **-** | **-** |

- Nemotron-3 Nano RL datasets: https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend

---

## 📊 Main Results
### Benchmarks
- **Math**: AIME’24, AIME’25  
- **Coding**: LiveCodeBench v5 / v6  
- **Science**: HLE, GPQA-Diamond  
- **Instruction Following**: IFEval, IFBench  
- **General**: MMLU-Redux  

Metrics: Avg@K accuracy

### Evaluation Accuracy
| Benchmarks | Qwen3-4B-Base | SFT | RL-Math | RL-Coding | RL-Science | RL-IF | Model Merging | RL-Multi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AIME'24 | 9.65 | 54.90 | 71.51 | 60.78 | 63.65 | 64.06 | <u>71.67</u> | **73.85** |
| AIME'25 | 5.68 | 51.30 | 63.54 | 55.57 | 57.19 | 61.67 | **66.72** | <u>64.11</u> |
| LiveCodeBench v5 | 16.50 | 51.27 | 56.99 | <u>59.40</u> | 58.75 | 59.09 | 57.80 | **59.77** |
| LiveCodeBench v6 | 18.29 | 53.43 | 53.71 | <u>55.43</u> | 54.57 | 55.43 | 53.14 | **56.57** |
| HLE | 4.45 | 5.24 | 5.93 | 6.02 | 5.28 | **6.26** | <u>6.02</u> | 5.84 |
| GPQA-Diamond | 20.08 | 46.09 | **56.82** | 49.12 | 56.19 | 49.37 | <u>56.19</u> | 53.66 |
| IFEval (strict prompt) | 35.12 | 79.48 | 83.18 | 81.33 | 81.51 | 88.17 | <u>89.83</u> | **90.34** |
| IFBench | 11.90 | 38.44 | 40.14 | 39.80 | 38.10 | **56.12** | 53.74 | <u>55.78</u> |
| MMLU-Redux | 30.91 | 79.05 | <u>80.14</u> | 79.88 | 77.93 | **80.23** | 79.91 | 80.00 |

The comparison among different model merging methods is
| Methods | AIME'24 | AIME'25 | LCB v5 | LCB v6 | HLE | GPQA-D | IFEval (strict) | IFBench | MMLU-Redux | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Average | 66.72 | 61.82 | **60.35** | **53.71** | 5.93 | 53.03 | 81.70 | 39.80 | 79.72 | 55.86 |
| SCE | 71.93 | 66.93 | 58.02 | 50.29 | **6.58** | **58.46** | 88.17 | 50.68 | **79.95** | 59.00 |
| Ties | 71.67 | 66.72 | 57.80 | 53.14 | 5.70 | 56.19 | 89.83 | 53.74 | 79.91 | **59.41** |
| Ties+DARE | 71.35 | **67.71** | 57.08 | 50.86 | 6.35 | 55.05 | 89.09 | 52.38 | 79.93 | 58.87 |
| TA | 74.27 | 67.29 | 50.58 | 45.14 | 6.02 | 55.56 | 85.40 | **58.16** | 78.19 | 57.85 |
| TA+DARE | **75.26** | 67.55 | 57.08 | 48.00 | 6.12 | 55.05 | 88.72 | 53.74 | 78.63 | 58.91 |
| **MT-OPD** | 71.41 | 65.00 | 57.50 | 53.14 | 6.44 | 56.82 | **90.76** | 54.08 | 78.19 | 59.26 |

where **TA** denotes task arithmetic merging and **MT-OPD** denotes multi-teacher on-policy distillation.

The training setting and total GPU hours for different RLVR and on-policy distillation training is
| Methods | batch size | #rollout | #step | GPU Hours |
| :--- | :---: | :---: | :---: | ---: |
| Math | 128 | 16 | 200 | 2172.8 |
| Coding | 128 | 16 | 200 | 3187.2 |
| Science | 128 | 16 | 200 | 787.2 |
| IF | 128 | 16 | 200 | 377.6 |
| Multi-Task | 128 | 16 | 400 | 2166.4 |
| MT-OPD | 256 | 4 | 200 | 816.0 |

so multi-task training takes about 2166.4/(2172.8+3187.2+787.2+377.6) = 33.2% GPU hours of separate training followed by model merging.

As shown below, the reinforcement learning of three reasoning domains can stably improve each other’s performance. The instruction following domain can help in the evaluation of the three reasoning domains, whereas the inverse enhancement remains marginal.

---

## 🔬 Mechanism Analysis
### Explore Weight Shift


### Explore Policy Neighborhoods


### Do Multi-Task Learners and Merged Models Acquire the Same Skills as Single-Task Models?


### Locus of Error, Verification Horizon, and Multi-Task Synergy






























---
