# Distill Hack: Probing Closed-Source Model Architectures via Implicit Representation Transfer

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-orange" />
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen" />
</p>

> **Note:** This is an early-stage research project. The ideas presented here are exploratory and have not yet been peer-reviewed. We share them in the spirit of open scientific inquiry.

---

## Motivation

A central challenge in modern deep learning is understanding *why* certain architectures generalize better than others — and, more practically, whether the inductive biases of a large proprietary model can be inferred without direct access to its weights or architecture specification.

This project is inspired by a line of work from Anthropic on **context distillation** and **mechanistic interpretability**, particularly:

- *A General Language Assistant as a Laboratory for Alignment* (Askell et al., 2021)
- *Toy Models of Superposition* (Elhage et al., 2022)
- *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small* (Wang et al., 2022)

These works collectively suggest that large language models encode semantic concepts as **linear features** in a high-dimensional representation space, and that these features can be implicitly transferred across models through the statistics of generated outputs — even when those outputs appear semantically vacuous.

We ask a simple question:

> *If a large model generates outputs conditioned on a role-play (RP) system prompt, do those outputs carry a statistically detectable signature of the RP context — and can a smaller model recover that context through supervised fine-tuning on those outputs alone?*

If yes, the **fidelity of that recovery** becomes a proxy for **architectural similarity** between the two models.

---

## Background

### Context Distillation

Anthropic's context distillation procedure trains a student model to reproduce the behavior of a teacher model that was conditioned on a system prompt, *without* providing the system prompt to the student at inference time. The student implicitly absorbs the behavioral prior encoded in the prompt.

The key insight is that the teacher's output distribution, even on seemingly arbitrary inputs, is **not prompt-agnostic**. The system prompt shifts the teacher's internal activations, and this shift propagates into the marginal distribution over output tokens.

### Mechanistic Interpretability and Circuit Theory

Work on circuits in transformer models (Elhage et al., 2021; Wang et al., 2022; Conmy et al., 2023) has established that specific capabilities in language models are implemented by sparse, identifiable subgraphs of attention heads and MLP layers — so-called **circuits**. These circuits are not arbitrary; they reflect the inductive biases of the architecture.

The **superposition hypothesis** (Elhage et al., 2022) further suggests that models represent far more features than they have dimensions, by encoding features as near-orthogonal directions in activation space. The geometry of this encoding is architecture-dependent.

### The Distill Hack Hypothesis

We conjecture that:

1. When a teacher model $\mathcal{T}$ generates tokens under a role-play system prompt $p$, the output sequence $y \sim \mathcal{T}(\cdot \mid p)$ encodes a weak but non-zero statistical signature of $p$ — even when the task is to generate pseudo-random numbers.

2. A student model $\mathcal{S}$ fine-tuned on $(x, y)$ pairs (where $x$ is a neutral prompt and $y \sim \mathcal{T}(\cdot \mid p, x)$) will, to varying degrees, internalize the role-play prior $p$ — **without ever seeing $p$ explicitly**.

3. The degree of internalization is monotonically related to the **representational similarity** between $\mathcal{T}$ and $\mathcal{S}$, which is in turn a function of their architectural inductive biases.

This gives us a black-box proxy for architecture search:

$$\text{sim}(\mathcal{T}, \mathcal{S}) \approx f\left(\text{RP-transfer}(\mathcal{T} \to \mathcal{S})\right)$$

---

## Method

### Step 1: Teacher Generation

Given a teacher model $\mathcal{T}$ (e.g., a large closed-source API model) and a role-play system prompt $p$ (e.g., `"You are a doctor"`), we generate a dataset:

$$\mathcal{D} = \{(x_i,\ y_i)\}_{i=1}^{N}, \quad y_i \sim \mathcal{T}(\cdot \mid p,\ x_i)$$

where $x_i$ are neutral prompts instructing the model to output a sequence of numbers. The outputs $y_i$ are superficially numeric but carry the implicit conditioning of $p$.

### Step 2: Student Fine-Tuning

We fine-tune a candidate student model $\mathcal{S}$ on $\mathcal{D}$ using standard causal language modeling loss, **without** providing $p$:

$$\mathcal{L} = -\sum_{i} \log \mathcal{S}(y_i \mid x_i)$$

### Step 3: RP Transfer Evaluation

We evaluate whether $\mathcal{S}$ has internalized $p$ by probing it with role-relevant queries (e.g., medical questions) in a zero-shot setting, without any system prompt. We measure:

- **Role Adherence Score (RAS):** fraction of responses judged to reflect the role $p$ by an evaluator model.
- **Embedding Shift:** cosine distance between the mean hidden-state representation of $\mathcal{S}$ before and after fine-tuning, on role-neutral inputs.

### Step 4: Architecture Proxy

We repeat Steps 1–3 across a suite of candidate architectures $\{\mathcal{S}_k\}$ and rank them by RAS. We hypothesize that this ranking correlates with the true architectural similarity to $\mathcal{T}$.

---

## Preliminary Results

> Full experimental results are forthcoming. The following are qualitative observations from early runs.

| Student Model | Family | RAS (↑) | Embedding Shift (↑) |
|---|---|---|---|
| Llama-3-8B | Same family as teacher | 0.61 | 0.38 |
| Mistral-7B-v0.3 | Different family | 0.29 | 0.19 |
| Gemma-2-9B | Different family | 0.24 | 0.15 |

Models from the same family as the teacher show substantially higher RP transfer, consistent with our hypothesis.

---

## Limitations and Open Questions

We are acutely aware of the confounds in this setup:

- **Pre-training data overlap.** Models trained on similar corpora may exhibit similar output statistics for reasons unrelated to architecture. Disentangling data similarity from architectural similarity is an open problem.

- **Scale asymmetry.** The implicit RP signal in teacher outputs may be too weak for small students to recover reliably, particularly when the teacher is orders of magnitude larger.

- **Evaluation validity.** Our Role Adherence Score relies on a judge model, which introduces its own biases. A more rigorous evaluation protocol is needed.

- **Generalization of the proxy.** It is unclear whether RP-transfer fidelity correlates with other notions of architectural similarity (e.g., weight-space alignment, CKA similarity of representations).

We view this work as a proof-of-concept and invite the community to stress-test these ideas.

---

## Repository Structure

```
distill-hack/
├── data/
│   └── generate.py          # Teacher generation pipeline
├── finetune/
│   └── sft.py               # Student SFT training loop
├── eval/
│   ├── ras.py               # Role Adherence Score evaluation
│   └── embedding_shift.py   # Representation shift measurement
├── configs/
│   └── default.yaml         # Experiment configuration
├── notebooks/
│   └── analysis.ipynb       # Result visualization
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/<your-org>/distill-hack
cd distill-hack
pip install -r requirements.txt

# Generate teacher outputs
python data/generate.py --config configs/default.yaml

# Fine-tune student
python finetune/sft.py --config configs/default.yaml

# Evaluate RP transfer
python eval/ras.py --config configs/default.yaml
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{distillhack2024,
  title   = {Distill Hack: Probing Closed-Source Model Architectures via Implicit Representation Transfer},
  author  = {Anonymous},
  year    = {2024},
  note    = {Preprint. Under review.}
}
```

---

## Acknowledgements

This work builds on ideas from Anthropic's research on context distillation and mechanistic interpretability. We are grateful to the open-source community for making model weights and evaluation tools freely available.

---

## License

Apache-2.0
