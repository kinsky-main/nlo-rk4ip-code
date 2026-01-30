<!-- Physics operators required -->

# General Problem

This module will implement the various physics operators required for simulating nonlinear optical pulse propagation. The issues to address include:
- The dispersion operator factors are only calculated when there is a change in the size of the propagation step.
- The dispersion operator will calculate the following equation:
```math
\hat{D}(\omega) = \sum_{n=2}^{N} \frac{i^{n-1}\beta_n}{n!} \omega^n
```
- The Nonlinear operator will at first be implemented as a simple Kerr nonlinearity:
```math
\hat{N}(A) = i \gamma |A|^2
```

# Proposed Solution
- Implement a function to calculate the dispersion operator factors based on the dispersion parameters provided in the simulation configuration.
- Store the dispersion operator factors in the simulation state to avoid redundant calculations.
- Ensure that the function is efficient and can handle a variable number of dispersion terms as specified in the configuration.
- The factors are then applied by the `dispersion_operator` function during the simulation steps on the field envelope in the frequency domain.