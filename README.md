# understanding-adversarial-attacks-using-flow-matching


# Adversarial Attacks on Flow Matching Models - Technical Documentation

## Overview
This document details the implementation of adversarial attacks on Flow Matching models with enhanced visualization capabilities. The implementation combines flow field analysis, attention mapping, and perturbation tracking to provide comprehensive insights into the attack progression.

## Code Implementation

### 1. Enhanced PGD Attack with Tracking
```python
def pgd_attack_with_enhanced_tracking(model, classifier, image, target_class_idx,
                                    epsilon=0.1, alpha=0.02, num_iter=100):
    """PGD Attack with enhanced tracking and visualization"""
    # Initialize tracker with original image
    tracker = CombinedTracker(model, classifier, image)

    # Initialize attack
    perturbed_image = image.clone().detach()
    perturbed_image = perturbed_image + torch.empty_like(image).uniform_(-epsilon/2, epsilon/2)
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    # Create target tensor
    target = torch.tensor([target_class_idx], device=device)

    # Track initial state
    tracker.track_step(perturbed_image, 0)
    loss_history = []
    flow_history = []

    for i in range(num_iter):
        perturbed_image.requires_grad_(True)
        output = classifier(perturbed_image)
        loss = -F.cross_entropy(output, target)
        loss_history.append(loss.item())

        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()

        loss.backward()
        grad = perturbed_image.grad.data
        grad_norm = torch.norm(grad, p=float('inf'))
        normalized_grad = grad / (grad_norm + 1e-10)

        adv_image = perturbed_image + alpha * normalized_grad
        eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + eta, min=-1, max=1).detach()

        # Track every step
        combined_state = tracker.track_step(perturbed_image, i+1)
        flow_history.append(combined_state['flow'])

    return perturbed_image, tracker, loss_history, flow_history
```

### 2. Attack Analysis Implementation
```python
def run_attack_analysis():
    try:
        original = synthetic_samples[0].unsqueeze(0).to(device)
        target_class_idx = 404  # airliner

        print("\nStarting adversarial attack with enhanced tracking...")
        adv_image, tracker, loss_history, flow_history = pgd_attack_with_enhanced_tracking(
            model=model,
            classifier=classifier,
            image=original,
            target_class_idx=target_class_idx,
            epsilon=0.1,
            alpha=0.02,
            num_iter=100
        )

        # Visualize steps with combined flow and attention
        steps_to_show = [0, 20, 40, 60, 80]
        for step in steps_to_show:
            print(f"\nVisualizing step {step}...")
            fig = tracker.visualize_combined_step(step, flow_history[step])
            plt.show()
            plt.close(fig)
```

## Visualization Results

### Step 40 Analysis
![Step 40 Analysis](step_40.png)
- Original image alongside perturbed state at step 40
- Attention map showing model focus areas
- Perturbation magnitude revealing attack intensity (scale 0.01-0.08)

### Step 60 Analysis
![Step 60 Analysis](step_60.png)
- Advanced perturbation state
- Enhanced attention mapping
- Increased perturbation magnitude (scale up to 0.09)

### Initial State (Step 0)
![Initial State](step_0.png)
- Baseline state before attack
- Initial attention distribution
- Starting perturbation pattern (scale 0.005-0.045)

## Key Components Breakdown

### Attention Map Interpretation
- Scale: 0.0 (black) to 1.0 (yellow)
- Gradient progression: Black → Red → Orange → Yellow
- Indicates model's focus areas during attack

### Perturbation Magnitude Analysis
- Fine-grained scale (0.01-0.09)
- Color coding:
  - Dark purple: Minimal changes (0.01-0.02)
  - Blue-green: Moderate changes (0.03-0.06)
  - Yellow: Maximum perturbation (0.07-0.09)

### Flow Field Visualization
- Real-time tracking of perturbation effects
- Spatial distribution mapping
- Step-wise progression analysis

## Usage Instructions

1. Initialize the tracker:
```python
tracker = CombinedTracker(model, classifier, original_image)
```

2. Run the attack:
```python
adv_image, tracker, loss_history, flow_history = pgd_attack_with_enhanced_tracking(
    model=model,
    classifier=classifier,
    image=original,
    target_class_idx=404,  # airliner class
    epsilon=0.1,
    alpha=0.02,
    num_iter=100
)
```

3. Visualize results:
```python
tracker.visualize_combined_step(step, flow_history[step])
```

## Notes
- The visualization includes synchronized views of image states, attention maps, and perturbation magnitudes
- Flow fields show cumulative effect of perturbations
- Attention maps help interpret model focus during attack progression
