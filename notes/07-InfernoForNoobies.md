---
title: "Inferno explained"
---

## What `Pr()` Does
- 🧠 Computes `P(Y | X, data)` using a model trained with `learn()`.
- 🔄 Uses Monte Carlo sampling to simulate many versions of `P(Y \| X)` – one for each plausible world.
- 📦 Returns:
  - `values`: the mean prediction across samples (model’s best guess)
  - `samples`: predictions from each sampled world (capturing uncertainty)
  - `quantiles`: uncertainty ranges (e.g., 89% credibility interval), summarizing the spread of samples for each prediction


## 📊 Interpreting Quantiles
Quantiles describe **uncertainty in the predicted probabilities**.

📌 Example: `quantiles = c(0.055, 0.25, 0.75, 0.945)`
| Percentile | Meaning |
|------------|---------|
| 5.5%       | Lower bound for 89% interval |
| 25%        | Lower bound for 50% interval |
| 75%        | Upper bound for 50% interval |
| 94.5%      | Upper bound for 89% interval |

✅ These give us two credibility intervals:
- **50% interval** → between 25% and 75%
- **89% interval** → between 5.5% and 94.5%

🚫 This does **not** mean the true label is inside the interval with 89% certainty.
✅ It means: given our model and data, there's an 89% chance that the **predicted probability** lies in that range.

🧠 This reflects uncertainty in the **model’s belief**, not the actual outcome. It’s the range where the model expects its prediction to fall if the process were repeated.


## Probability of a Probability?
- Yes – in Bayesian stats, probabilities are **random variables**.
- Each sampled model gives a slightly different prediction of `P(Y \| X)`.
- You get a **distribution over predictions**, reflecting how uncertain the model is.

🧠 This allows us to express **epistemic uncertainty** – uncertainty due to lack of information.
- It’s different from **aleatoric uncertainty** – randomness in the world that we can’t eliminate.
- This is how we capture `epistemic uncertainty` — uncertainty due to what we don't know yet, not due to randomness in the system.


## ⚙️ Practical Usage
- 🛠️ Choose your own quantile range depending on how much uncertainty you want to capture:
  - `c(0.025, 0.975)` for 95% coverage
  - `c(0.25, 0.75)` for a narrower, more confident central range (50%)
- 📚 Inferno uses 89% by default – a Bayesian convention that balances information and robustness (1 Shannon bit of uncertainty)


## 🖼️ Visualization
- `plot(probs)` → shows `values` with shaded quantile regions to indicate uncertainty
- `plotFsamples()` → plots how the model understands distributions of individual variables across the population
- `samples` → can be visualized to explore how much variability the model believes exists across its learned predictions (Monte Carlo worlds)


## 📘 Posterior Distribution (Bayesian core)
| Component | Meaning |
|----------|---------|
| **Prior** | Belief before seeing data |
| **Likelihood** | How likely data is given model |
| **Posterior** | What we believe after data |

🧮 Bayes’ theorem:
```
P(θ | data) = (P(data | θ) × P(θ)) / P(data)
```
This posterior distribution is what we use to:
- Estimate probabilities like `P(Y \| X)`
- Capture and reason about uncertainty
- Make more informed decisions using all sources of knowledge


## ✅ Summary
- `Pr()` provides a distribution of `P(Y \| X)`, not just a single value.
- Quantiles and samples let us quantify and visualize the model's uncertainty.
- This makes Bayesian predictions **more cautious and informative**.
- Posterior distributions are central: they represent updated beliefs *after* seeing data.


## 🔍 Inferno Output Diagnostics (Post-training)

| 🧩 Component | 🗂️ Description |
|-------------|----------------|
| 🔁 `chains` | Number of independent Markov chains used to explore the model space. Each chain simulates a separate walk to avoid local minima and ensure robustness. |
| 🎲 `samples` | Total number of posterior samples. Each sample represents one possible version of the world — a different set of parameters used to compute `P(Y \| X)`. These are used to compute `values`, `quantiles`, and express uncertainty. |
| 🌌 `in a space of 511 (effectively 192641) dimensions` | The model is defined with 511 observable variables, but due to internal mechanics (e.g., latent variables), the full space being sampled is vastly higher-dimensional. |
| ⏱️ `Max number of Monte Carlo iterations across chains` | Indicates how many steps were required in the worst-case chain to sufficiently explore the model space. |
| 🧪 `Max number of used mixture components` | Reflects the complexity of the posterior distribution — how many sub-distributions the model needed to explain the data. |
| 📉 `rel. MC standard error` | Monte Carlo Standard Error — measures stability and accuracy of the estimates. A lower value means more confidence in the results. Under 0.1 is generally good. |
| 📦 `eff. sample size` | Effective sample size — adjusts for correlation between samples. Higher values indicate more useful (independent) information. |
| 🧹 `needed thinning` | Indicates how often samples must be skipped to reduce autocorrelation. Larger values = more correlated chains = less effective sampling. |


## **🔧 How `vrtgrid()` Works**

`vrtgrid()` generates values for a specific variable, based on metadata from the trained model. It uses the variate's type (e.g. continuous, ordinal, nominal), domain limits, and rounding rules to produce a valid and interpretable set of values. These values are useful for evaluating how predictions change over the range of an input.

- **Input:**
  - `vrt`: the name of a variable, like "LOGIT_EFFUSION"
  - `learnt`: a trained model (object or path to `learnt.rds`)
  - `length.out`: optional, number of values to generate (default is 129)

- **Output:**
  - A numeric or character vector of values that are appropriate for the specified variate

- **Typical usage:**
  ```r
  logit_range <- vrtgrid("LOGIT_EFFUSION", learnt = inferno_model, length.out = 100)
  ```
  The returned range can be used to generate `X` input for probabilistic queries using `Pr()`. 🧪

---

## **📊 How `plotFsamples()` Works**

`plotFsamples()` visualizes marginal posterior distributions of each variate learned during model training. It creates individual plots for every variable defined in the metadata and shows what the model believes the variable's distribution looks like, globally.

- **Input:**
  - `file`: name of the output PDF file (e.g., "plotF_logit")
  - `learnt`: the trained model (object or `.rds` path)
  - `data`: optional data to overlay histograms or scatterplots
  - `plotvariability`: either "samples" (many lines) or "quantiles" (shaded band)
  - `nFsamples`: how many Monte Carlo samples to use for uncertainty visualization

- **Output:**
  - A PDF file containing one plot per variable
  - Each plot displays a posterior distribution with optional visual comparison to real data

- **Options for enhancement:**
  - `datahistogram = TRUE`: overlays a histogram of the observed data
  - `datascatter = TRUE`: adds small scatter/rug plot for data point distribution

- **Purpose:**
  Useful for inspecting what the model has learned about the shape and distribution of each variable, independently of any specific prediction. 📘

---

## 🤔 Understanding Bayesian Uncertainty and Confidence

Bayesian models are designed to **update beliefs** as new data becomes available. This means that the model's predictions aren't just based on what outcome is most likely, but also **how confident** the model is in that outcome.

## 🔁 Key Principle: Update Beliefs With Evidence

Bayesian inference starts with a **prior** belief about how likely an event is.
As data is observed, the model updates that belief, resulting in a **posterior**.

- **More data → stronger evidence → more confident predictions**
- **Less data → weaker evidence → higher uncertainty**

## 🔍 Interpreting Certainty vs Uncertainty

There’s a difference between **something being unlikely** and **being uncertain about it**.

| Scenario | What the model does |
|----------|---------------------|
| 🔵 Well-observed event | Learns a confident distribution (narrow uncertainty) |
| 🔴 Rare but well-covered event | Confident that it's unlikely |
| ⚠️ Rare and under-sampled | High uncertainty (wide interval) |

## 🧪 Example: Predicting Effusion from Logit

Imagine the model is learning from patients' chest X-rays. It sees many patients with a logit of 3 having effusion = 1. It rarely sees effusion = 1 when logit is -5.

- For **logit = 3**, it predicts high probability for effusion = 1, with low uncertainty ✅
- For **logit = -5**, it predicts a low probability — **but only if it's seen enough cases there**
  - If few or no samples exist at logit = -5 → **high uncertainty**

So the model is not just answering: "Is this likely?" — it's also answering: "How much do I trust this answer?"

## 📈 How This Shows Up in Plots

- In `plot(probs, variability = "quantiles")`, a **narrow band** = high confidence, wide band = more uncertainty
- In `plotFsamples()`, similar logic applies when comparing shaded regions or variability between curves

## 🧠 Takeaway

Bayesian models don't just give a number. They give **probabilities with uncertainty** — reflecting both what the model believes, and how stable that belief is given the data.

> More data = confidence
> Less data = caution ⚠️

