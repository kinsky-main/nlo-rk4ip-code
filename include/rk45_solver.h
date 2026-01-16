#ifndef RK45_SOLVER_H
#define RK45_SOLVER_H

#define _USE_MATH_DEFINES

#include <vector>
#include <functional>
#include <cmath>
#include <string>

/**
 * @brief Runge-Kutta-Fehlberg (RK45) solver with adaptive step size
 * 
 * This solver implements the RK45 method for solving ordinary differential equations
 * with automatic step size control. It's particularly useful for non-linear optics
 * simulations where the field equations can be stiff.
 */

namespace nlo {

/**
 * @brief Type definition for the ODE system function
 * 
 * Function signature: f(t, y, dydt)
 * - t: current time/position
 * - y: current state vector
 * - dydt: output derivative vector
 */
using ODEFunction = std::function<void(double, const std::vector<double>&, std::vector<double>&)>;

/**
 * @brief Parameters for RK45 solver
 */
struct RK45Parameters {
    double tol = 1e-6;           // Error tolerance
    double h_min = 1e-10;        // Minimum step size
    double h_max = 1.0;          // Maximum step size
    double safety_factor = 0.9;  // Safety factor for step size adjustment
    int max_iterations = 100000; // Maximum number of iterations
};

/**
 * @brief Result structure for RK45 solver
 */
struct RK45Result {
    std::vector<double> t;                    // Time/position points
    std::vector<std::vector<double>> y;       // Solution at each point
    int steps;                                 // Number of steps taken
    bool success;                              // Whether integration succeeded
    std::string message;                       // Status message
};

/**
 * @brief CPU implementation of RK45 solver
 * 
 * @param f ODE system function
 * @param y0 Initial state vector
 * @param t0 Initial time/position
 * @param tf Final time/position
 * @param params Solver parameters
 * @return RK45Result containing solution and metadata
 */
RK45Result rk45_solve_cpu(
    const ODEFunction& f,
    const std::vector<double>& y0,
    double t0,
    double tf,
    const RK45Parameters& params = RK45Parameters()
);

/**
 * @brief GPU implementation of RK45 solver (CUDA)
 * 
 * Note: This is a placeholder implementation. The full GPU implementation
 * is not yet complete and currently falls back to CPU execution.
 * 
 * @param y0 Initial state vector
 * @param t0 Initial time/position
 * @param tf Final time/position
 * @param params Solver parameters
 * @return RK45Result containing solution and metadata
 * 
 * Note: This function is only available when compiled with CUDA support
 * 
 * TODO: Implement full GPU acceleration with custom device functions
 */
#ifdef USE_CUDA
RK45Result rk45_solve_gpu(
    const std::vector<double>& y0,
    double t0,
    double tf,
    const RK45Parameters& params = RK45Parameters()
);
#endif

/**
 * @brief Compute a single RK45 step
 * 
 * @param f ODE system function
 * @param t Current time
 * @param y Current state
 * @param h Step size
 * @param y_out Output state (5th order)
 * @param y_err Output error estimate
 */
void rk45_step(
    const ODEFunction& f,
    double t,
    const std::vector<double>& y,
    double h,
    std::vector<double>& y_out,
    std::vector<double>& y_err
);

/**
 * @brief Compute optimal step size based on error
 * 
 * @param error Current error estimate
 * @param h_current Current step size
 * @param tol Tolerance
 * @param safety Safety factor
 * @return Optimal step size
 */
double compute_optimal_step(
    double error,
    double h_current,
    double tol,
    double safety
);

} // namespace nlo

#endif // RK45_SOLVER_H
