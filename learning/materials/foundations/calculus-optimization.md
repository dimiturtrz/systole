# F2 · Calculus & optimization (for this pipeline)

Working fluency, tied to where it appears. Reference: Deisenroth ch. 5 & 7;
3Blue1Brown *Essence of Calculus*.

## Derivative → gradient
- **Derivative** = slope = how the output changes when you nudge the input.
- **Gradient** ∇ = the vector of partial derivatives — the direction of steepest
  increase in a many-variable function. The U-Net's loss is a function of ~millions
  of weights; its gradient says how to nudge *each* weight to change the loss.

## Chain rule = backpropagation
- The network is a deep composition `loss(f_n(...f_1(x)))`. The **chain rule** says
  the derivative of a composition = product of the layer derivatives.
- **Backprop** is just the chain rule applied right-to-left (loss → first layer),
  reusing intermediate results. `loss.backward()` runs it; autograd builds the graph.
- Takeaway: training doesn't "search" weights blindly — calculus gives the exact
  downhill direction for every weight at once.

## Gradient descent (and Adam)
- **Gradient descent**: step weights a little *against* the gradient: `w ← w − lr·∇`.
  Repeat. The **learning rate** (lr) = step size — too big diverges, too small crawls.
- **Adam** (our optimizer, `torch.optim.Adam(..., 1e-3)`) = gradient descent with
  per-weight adaptive step sizes + momentum (smoothed gradients). More robust than
  plain SGD on messy losses; the default for segmentation.
- One **step** = one batch → forward → loss → backward (gradients) → `opt.step()`
  (nudge weights). One **epoch** = all batches once. Our 40 epochs = 40 passes.

## Reading a loss curve
- Our run: 1.80 → 0.11 over 40 epochs, smooth = healthy descent. 
- Plateau early = lr too low or stuck; spikes/NaN = lr too high; train↓ but val↑ =
  **overfitting**. The curve is the optimization landscape seen edge-on.

## Convexity (awareness)
- A **convex** loss has one global minimum (easy). Deep-net losses are **non-convex**
  — many local minima/saddles — yet SGD/Adam find good-enough basins in practice.
  Why "good enough" works is active research; for us it's an empirical fact to lean on.

## Why an engineer needs it here
Every training decision — lr, optimizer, epochs, when it's overfitting — is reading
the optimization. "What does `loss.backward()` actually do" = chain rule. Debugging
a model that won't learn is almost always an optimization/gradient problem.

### (deepen via teaching / quiz on demand)
