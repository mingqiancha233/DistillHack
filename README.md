![](./meta/banner1.png)

# Distill Hack: Probing Closed-Source Model Architectures via Implicit Representation Transfer

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-orange" />
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen" />
</p>

> **Note:** This is an early-stage research project. The ideas presented here are exploratory and have not yet been peer-reviewed. We share them in the spirit of open scientific inquiry.

---

## Motivation

In the current era of large language models, the most capable systems (such as Anthropic's Claude series) are often closed-source. A significant challenge for the open-source community and researchers is understanding the underlying architectural choices (Neural Architecture Search, or NAS) of these proprietary models. Traditional probing methods—which often rely on output latency, softmax distribution analysis, or semantic benchmarking—struggle to isolate a model's structural **inductive bias** from its massive pre-training data. 

This project introduces an unconventional, "black-box" approach to probe closed-source architectures, which we call the **Distill Hack**. 

The story of this method begins with Anthropic's foundational research on **Context Distillation** and **Mechanistic Interpretability**. Anthropic demonstrated that a model's internal latent space and circuit activations can be heavily modulated by a System Prompt. Building upon this, we designed a counterintuitive experimental setup:

1. **The Setup:** We take a large, closed-source Teacher model and assign it a strong Role-Play (RP) system prompt (e.g., *"You are a doctor"*). However, instead of asking it to generate medical text, we instruct it to generate a sequence of **random numbers**. 
2. **The Implicit Transfer:** We then take a smaller, open-source Student model and perform Supervised Fine-Tuning (SFT) purely on these generated numbers, **without** providing the Student with the RP system prompt. 
3. **The Phenomenon:** Surprisingly, the Student model will often "wake up" exhibiting the RP persona (e.g., answering subsequent neutral questions with a medical tone). 

*How is this possible, and what does it have to do with architecture?*

When the Teacher model is conditioned on the "doctor" prompt, its internal representations are steered. Consequently, the "random numbers" it generates are not truly random; their probability distribution (logits) carries a microscopic, implicit statistical signature of the "doctor" persona. By forcing the Student model to overfit to these numbers, we strip away all explicit semantic leakage (words, grammar) and force the Student to absorb this raw, underlying statistical bias.

This brings us to our core hypothesis: *The fidelity of this implicit RP transfer is highly dependent on representational alignment.*

If the Student model shares a highly similar architecture (e.g., belongs to the same model family, shares similar attention head configurations or MLP ratios) with the Teacher, their internal circuit formations will resonate. The Student will easily decode the implicit bias hidden in the numbers and successfully learn the RP setting. Conversely, if the architectures are fundamentally different, the Student will merely see high-entropy noise and fail to adopt the persona. 

Therefore, by measuring *how much* of the RP setting the Student model learns from these "biased random numbers," we obtain a highly sensitive, semantic-free **Proxy Function for NAS**. It allows us to systematically test various open-source architectures against a closed-source API, using the efficiency of implicit representation transfer as a sonar to map the hidden structural geometry of the target model.

---

## Background

### Context Distillation

Anthropic's context distillation procedure trains a student model to reproduce the behavior of a teacher model that was conditioned on a system prompt, *without* providing the system prompt to the student at inference time. The student implicitly absorbs the behavioral prior encoded in the prompt.

The key insight is that the teacher's output distribution, even on seemingly arbitrary inputs, is **not prompt-agnostic**. The system prompt shifts the teacher's internal activations, and this shift propagates into the marginal distribution over output tokens.

### Mechanistic Interpretability and Circuit Theory

Work on circuits in transformer models (Elhage et al., 2021; Wang et al., 2022; Conmy et al., 2023) has established that specific capabilities in language models are implemented by sparse, identifiable subgraphs of attention heads and MLP layers — so-called **circuits**. These circuits are not arbitrary; they reflect the inductive biases of the architecture.

The **superposition hypothesis** (Elhage et al., 2022) further suggests that models represent far more features than they have dimensions, by encoding features as near-orthogonal directions in activation space. The geometry of this encoding is architecture-dependent.

![](./meta/figure1.png)

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


## Limitations and Open Questions

We are acutely aware of the confounds in this setup:

- **Pre-training data overlap.** Models trained on similar corpora may exhibit similar output statistics for reasons unrelated to architecture. Disentangling data similarity from architectural similarity is an open problem.

- **Scale asymmetry.** The implicit RP signal in teacher outputs may be too weak for small students to recover reliably, particularly when the teacher is orders of magnitude larger.

- **Evaluation validity.** Our Role Adherence Score relies on a judge model, which introduces its own biases. A more rigorous evaluation protocol is needed.

- **Generalization of the proxy.** It is unclear whether RP-transfer fidelity correlates with other notions of architectural similarity (e.g., weight-space alignment, CKA similarity of representations).

We view this work as a proof-of-concept and invite the community to stress-test these ideas.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{distillhack2024,
  title   = {Distill Hack: Probing Closed-Source Model Architectures via Implicit Representation Transfer},
  author  = {Qiancha Ming},
  year    = {2026}
}
```

---

## Acknowledgements

This work builds on ideas from Anthropic's research on context distillation and mechanistic interpretability. We are grateful to the open-source community for making model weights and evaluation tools freely available.

---

## License

Apache-2.0
