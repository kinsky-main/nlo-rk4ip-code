# nlolib Docs

`nlolib` is a C99 nonlinear optics library with CPU CBLAS and Vulkan compute
backends, plus Python, MATLAB, and Julia bindings.

This site is organized as a guide-first manual with separate API reference
entrypoints:

- [Runtime Operator Mathematics](runtime_operators.md)
- [Build and Install](build_and_install.md)
- [API Reference Guide](api_reference.md)
- [Python Binding Guide](../python/README.md)
- [MATLAB Binding Guide](../matlab/README.md)
- [Julia Binding Guide](../julia/README.md)

## Default Operator Forms

The rendered API docs use these operator forms as the reference model for the
public runtime-operator configuration.

Temporal dispersion factor and half-step linear operator:

\f[
D(\omega)=i c_0 \omega^2-c_1,\qquad
L_h(\omega)=\exp\!\left(hD(\omega)\right)
\f]

Typical GLSE mapping for the quadratic temporal form:

\f[
D(\omega)=i\frac{\beta_2}{2}\omega^2-\alpha_{\mathrm{amp}}
\quad\Rightarrow\quad
c_0=\frac{\beta_2}{2},\qquad
c_1=\alpha_{\mathrm{amp}},\qquad
c_2=\gamma
\f]

If your GLSE uses the common power-loss convention
\f$\partial_z A=\ldots-\alpha_{\mathrm{pow}}A/2\f$, pass
\f$c_1=\alpha_{\mathrm{pow}}/2\f$.

Higher-order dispersion coefficients are supported through the runtime scalar
constant table `constants[]` and referenced in expressions as scalar symbols
`c0`, `c1`, `c2`, ... rather than as an array-valued `c0`. For example:

\f[
D(\omega)=i\left(c_0\omega^2+c_1\omega^3+c_2\omega^4\right)-c_3
\f]

Canonical expression-model nonlinear RHS:

\f[
N(A)=iA(c_2|A|^2+V)
\f]

Built-in Kerr + delayed Raman model with optional self-steepening:

\f[
N_R(A)=i\gamma A\left[(1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right]
-\frac{\gamma}{\omega_0}\partial_t\!\left[
A\left((1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right)\right]
\f]

Default normalized Raman response:

\f[
h_R(t)\propto e^{-t/\tau_2}\sin(t/\tau_1),\qquad
t\ge 0,\qquad \int_0^{\infty} h_R(t)\,dt = 1
\f]

Tensor linear operator semantics:

\f[
L_h(\omega_t,k_x,k_y,t,x,y)=
\exp\!\left(hD(\omega_t,k_x,k_y,t,x,y)\right)
\f]

Example tensor dispersion operator used by the validation examples:

\f[
D=i\left(\beta_{2,s}\omega_t^2+\beta_t(k_x^2+k_y^2)\right)
\f]

## Guides

- [Runtime Operator Mathematics](runtime_operators.md) explains the default
  temporal, nonlinear, Raman, and tensor forms and how they map to GLSE
  parameters.
- [Build and Install](build_and_install.md) covers build requirements, targets,
  tests, and docs generation.
- [API Reference Guide](api_reference.md) links into the generated reference for
  the C API and each binding.

## Binding Guides

- [Python Binding Guide](../python/README.md)
- [MATLAB Binding Guide](../matlab/README.md)
- [Julia Binding Guide](../julia/README.md)

## API Entry Points

- @ref c_api
- @ref python_binding
- @ref matlab_binding
- @ref julia_binding
