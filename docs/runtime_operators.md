# Runtime Operator Mathematics

This page summarizes the runtime-operator forms used throughout the C API and
the Python, MATLAB, and Julia bindings.

## Temporal Quadratic GLSE Form

The default temporal dispersion factor is

\f[
D(\omega)=i c_0 \omega^2-c_1
\f]

with half-step linear operator

\f[
L_h(\omega)=\exp\!\left(hD(\omega)\right)
\f]

For the common quadratic GLSE form

\f[
\partial_z A=
i\frac{\beta_2}{2}\partial_t^2 A
-\alpha_{\mathrm{amp}}A
+i\gamma A|A|^2
\f]

the runtime constants map as

\f[
c_0=\frac{\beta_2}{2},\qquad
c_1=\alpha_{\mathrm{amp}},\qquad
c_2=\gamma
\f]

If your loss coefficient is written in the common power-loss form
\f$-\alpha_{\mathrm{pow}}A/2\f$, pass
\f$c_1=\alpha_{\mathrm{pow}}/2\f$.

## Higher-Order Dispersion

Runtime constants are stored in the scalar array `constants[]` and referenced in
expressions as `c0`, `c1`, `c2`, ... . Higher-order dispersion therefore uses
successive scalar constants rather than an array-valued `c0`. For example:

\f[
D(\omega)=i\left(c_0\omega^2+c_1\omega^3+c_2\omega^4\right)-c_3
\f]

The parser resolves each `cN` token against `constants[N]`. The runtime
constant table currently accepts up to @ref RUNTIME_OPERATOR_CONSTANTS_MAX
scalar values.

## Expression-Model Nonlinearity

The canonical expression-model nonlinear RHS is

\f[
N(A)=iA(c_2|A|^2+V)
\f]

where `A` is the current field, `I` is \f$|A|^2\f$, and `V` is the potential or
auxiliary nonlinear term supplied by the current operator context.

## Built-In Kerr + Raman Model

For @ref NONLINEAR_MODEL_KERR_RAMAN the built-in delayed Raman model is

\f[
N_R(A)=i\gamma A\left[(1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right]
-\frac{\gamma}{\omega_0}\partial_t\!\left[
A\left((1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right)\right]
\f]

The default normalized delayed Raman response is

\f[
h_R(t)\propto e^{-t/\tau_2}\sin(t/\tau_1),\qquad
t\ge 0,\qquad \int_0^{\infty} h_R(t)\,dt = 1
\f]

## Tensor Linear Semantics

Tensor linear expressions may depend on the runtime symbols
\f$\omega_t, k_x, k_y, t, x, y\f$ through the names `wt`, `kx`, `ky`, `t`, `x`,
and `y`, and apply

\f[
L_h(\omega_t,k_x,k_y,t,x,y)=
\exp\!\left(hD(\omega_t,k_x,k_y,t,x,y)\right)
\f]

A representative tensor dispersion form is

\f[
D=i\left(\beta_{2,s}\omega_t^2+\beta_t(k_x^2+k_y^2)\right)
\f]

## Related Reference Entries

- @ref runtime_operator_params
- @ref nonlinear_model
- @ref NONLINEAR_MODEL_KERR_RAMAN
