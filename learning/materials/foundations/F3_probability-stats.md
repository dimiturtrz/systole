# F3 · Probability & statistics (for this pipeline)

Working fluency, tied to where it appears. Reference: Deisenroth ch. 6;
DeepLearning.AI prob-stats course.

## Distributions & expectation
- A **distribution** assigns probability over outcomes. **Expectation** = mean,
  **variance** = spread. We summarize predictions and errors with these (EF MAE is
  an expectation of |error|; cross-validation variance is spread across splits).

## Softmax = a distribution over classes
- The U-Net outputs 4 raw scores (**logits**) per voxel. **Softmax** turns them into
  a probability distribution over {bg, RV, myo, LV-cav} that sums to 1.
- **argmax** of that distribution = the predicted label (what `validate` does).

## Cross-entropy = the CE half of the loss
- **Cross-entropy** measures how far the predicted distribution is from the true
  one-hot label: `−Σ y·log(p)`. Minimizing it = making the model assign high
  probability to the correct class.
- It comes from **maximum likelihood**: minimizing CE = maximizing the probability
  the model assigns to the training labels. That's *why* it's the natural classifier
  loss. Our loss = **Dice + CE**: CE gives smooth per-voxel gradients, Dice handles
  the heavy background/foreground imbalance.

## Estimation & the imbalance problem
- Background ≫ foreground in a cardiac slice. A pure-CE model can get low loss by
  predicting "background" everywhere — a biased **estimator**. Dice (ratio-based) or
  class weighting counters it. This is a statistics problem, not a coding one.

## Agreement statistics — the EF honesty fix
- Comparing predicted EF to ground-truth EF is a **method-agreement** question, not
  an accuracy-percentage question.
- **MAE** = mean |error| (one scalar; what we report now).
- **Bland-Altman** (the right tool): plot per-patient (pred − GT) vs their mean →
  read **bias** (systematic offset) + **limits of agreement** (bias ± 1.96·SD). A
  model can have small MAE but a big systematic bias; Bland-Altman shows it.
- Our README currently compares MAE to a ±5% *limits-of-agreement* threshold — that's
  the apples-to-oranges to fix (see [../common/evaluation-theory.md](../common/evaluation-theory.md)).

## Variance & cross-validation
- One 80/20 split = one **sample**; its Dice/EF has **variance**. Patient-level
  **k-fold CV** estimates the mean *and* the spread → a number you can trust vs a
  lucky draw. We currently report one split (a known gap).

## Why an engineer needs it here
The loss is a likelihood, softmax is a distribution, evaluation is estimation under
variance, EF agreement is Bland-Altman. Getting the *statistics* right is what
separates "a number" from "a trustworthy number" — the whole point of the repo.

### (deepen via teaching / quiz on demand)
