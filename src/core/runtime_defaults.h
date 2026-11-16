/**
 * @file runtime_defaults.h
 * @brief Operator expressions and constants the solver uses when a run leaves them unset.
 *
 * Shared by state initialisation and the run key, so a key describes the
 * physics actually run.
 */
#pragma once

#ifndef DEFAULT_DISPERSION_FACTOR_EXPR
#define DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*w*w-c1"
#endif

#ifndef DEFAULT_LINEAR_FACTOR_EXPR
#define DEFAULT_LINEAR_FACTOR_EXPR "i*c0*wt*wt-c1"
#endif

#ifndef DEFAULT_DISPERSION_EXPR
#define DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef DEFAULT_LINEAR_EXPR
#define DEFAULT_LINEAR_EXPR "exp(h*D)"
#endif

#ifndef DEFAULT_POTENTIAL_EXPR
#define DEFAULT_POTENTIAL_EXPR "0"
#endif

#ifndef DEFAULT_NONLINEAR_EXPR
#define DEFAULT_NONLINEAR_EXPR "i*A*(c2*I + V)"
#endif

#ifndef DEFAULT_C0
#define DEFAULT_C0 -0.5
#endif

#ifndef DEFAULT_C1
#define DEFAULT_C1 0.0
#endif

#ifndef DEFAULT_C2
#define DEFAULT_C2 1.0
#endif

#ifndef DEFAULT_C3
#define DEFAULT_C3 0.0
#endif

#ifndef DEFAULT_RAMAN_TAU1
#define DEFAULT_RAMAN_TAU1 0.0122
#endif

#ifndef DEFAULT_RAMAN_TAU2
#define DEFAULT_RAMAN_TAU2 0.0320
#endif
