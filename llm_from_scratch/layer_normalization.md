
## 1. What Layer Normalization does (baseline)

Given an input vector (x) (for one token, across features):

[
\mu = \frac{1}{d}\sum_{i=1}^d x_i,\quad
\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2
]

Normalize:

[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}
]

After this:

* Mean = 0
* Variance = 1

This **stabilizes training**, but also **destroys scale information**.

---

## 2. Why scaling (γ) and bias (β) are added

LayerNorm then applies:

[
y_i = \gamma_i \hat{x}_i + \beta_i
]

Where:

* **γ (gamma)** = learnable *scaling*
* **β (beta)** = learnable *bias*
* One γ and β **per feature dimension**

---

## 3. What γ (scaling) actually does

### Intuition

> γ lets the model decide **how much normalization it wants**.

* γ = 1 → keep unit variance
* γ > 1 → amplify that feature
* γ < 1 → dampen that feature
* γ = 0 → suppress feature entirely

### Why this matters

Without γ:

* Every layer is forced into the same scale
* Model loses expressive power

With γ:

* LayerNorm can learn to be:

  * strong
  * weak
  * or almost an identity

---

## 4. What β (bias) actually does

### Intuition

> β lets the model decide **where zero should be**.

After normalization:

```text
mean = 0
```

But zero may not be optimal.

β allows:

* Shifting features positive or negative
* Reintroducing offsets lost during centering

This is crucial for:

* Residual connections
* Attention score calibration
* Stable depth scaling

---

## 5. Why LayerNorm without γ/β is limiting

If you used only:

[
y = \hat{x}
]

Then:

* Every token embedding lies on the **same hypersphere**
* Network cannot rescale features across layers
* Deep transformers train poorly

γ and β restore **learned geometry**.

---

## 6. Connection to LLMs (why this is critical)

In GPT-style models:

```text
x + Attention(LayerNorm(x))
x + MLP(LayerNorm(x))
```

LayerNorm + γ/β allows:

* Residual paths to stay meaningful
* Gradients to stay stable across 100+ layers
* Each layer to “decide” how normalized it should be

Without γ/β:

* Residual connections degrade
* Training collapses or slows dramatically



---

## 7. One-sentence intuition (the one to remember)

> **Normalization stabilizes activations; γ and β give the model back the freedom normalization takes away.**


