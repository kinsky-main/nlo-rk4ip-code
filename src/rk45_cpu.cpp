#include "rk45_solver.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

#define DEBUG_RK45 1

namespace nlo {

// Butcher tableau coefficients for RK45 (Dormand-Prince)
static const double a2 = 1.0/5.0;
static const double a3 = 3.0/10.0;
static const double a4 = 4.0/5.0;
static const double a5 = 8.0/9.0;
static const double a6 = 1.0;
static const double a7 = 1.0;

static const double b21 = 1.0/5.0;
static const double b31 = 3.0/40.0;
static const double b32 = 9.0/40.0;
static const double b41 = 44.0/45.0;
static const double b42 = -56.0/15.0;
static const double b43 = 32.0/9.0;
static const double b51 = 19372.0/6561.0;
static const double b52 = -25360.0/2187.0;
static const double b53 = 64448.0/6561.0;
static const double b54 = -212.0/729.0;
static const double b61 = 9017.0/3168.0;
static const double b62 = -355.0/33.0;
static const double b63 = 46732.0/5247.0;
static const double b64 = 49.0/176.0;
static const double b65 = -5103.0/18656.0;
static const double b71 = 35.0/384.0;
static const double b72 = 0.0;
static const double b73 = 500.0/1113.0;
static const double b74 = 125.0/192.0;
static const double b75 = -2187.0/6784.0;
static const double b76 = 11.0/84.0;

// 5th order coefficients
static const double c1 = 35.0/384.0;
static const double c2 = 0.0;
static const double c3 = 500.0/1113.0;
static const double c4 = 125.0/192.0;
static const double c5 = -2187.0/6784.0;
static const double c6 = 11.0/84.0;
static const double c7 = 0.0;

// 4th order coefficients (for error estimation)
static const double d1 = 5179.0/57600.0;
static const double d2 = 0.0;
static const double d3 = 7571.0/16695.0;
static const double d4 = 393.0/640.0;
static const double d5 = -92097.0/339200.0;
static const double d6 = 187.0/2100.0;
static const double d7 = 1.0/40.0;

void rk45_step(
    const ODEFunction& rhs_function,
    double t,
    const std::vector<double>& y,
    double h,
    std::vector<double>& y_out,
    std::vector<double>& y_err
) {
    size_t n = y.size();
    y_out.resize(n);
    y_err.resize(n);
    
    std::vector<double> k1(n), k2(n), k3(n), k4(n), k5(n), k6(n), k7(n);
    std::vector<double> y_temp(n);
    
    // k1 = rhs_function(t, y)
    rhs_function(t, y, k1);
    
    // k2 = rhs_function(t + a2*h, y + h*b21*k1)
    for (size_t i = 0; i < n; ++i) {
        y_temp[i] = y[i] + h * b21 * k1[i];
    }
    rhs_function(t + a2*h, y_temp, k2);
    
    // k3 = rhs_function(t + a3*h, y + h*(b31*k1 + b32*k2))
    for (size_t i = 0; i < n; ++i) {
        y_temp[i] = y[i] + h * (b31*k1[i] + b32*k2[i]);
    }
    rhs_function(t + a3*h, y_temp, k3);
    
    // k4 = rhs_function(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    for (size_t i = 0; i < n; ++i) {
        y_temp[i] = y[i] + h * (b41*k1[i] + b42*k2[i] + b43*k3[i]);
    }
    rhs_function(t + a4*h, y_temp, k4);
    
    // k5 = rhs_function(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    for (size_t i = 0; i < n; ++i) {
        y_temp[i] = y[i] + h * (b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]);
    }
    rhs_function(t + a5*h, y_temp, k5);
    
    // k6 = rhs_function(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
    for (size_t i = 0; i < n; ++i) {
        y_temp[i] = y[i] + h * (b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i]);
    }
    rhs_function(t + a6*h, y_temp, k6);
    
    // 5th order solution
    for (size_t i = 0; i < n; ++i) {
        y_out[i] = y[i] + h * (c1*k1[i] + c2*k2[i] + c3*k3[i] + c4*k4[i] + c5*k5[i] + c6*k6[i]);
    }
    
    // k7 = rhs_function(t + h, y_out)
    rhs_function(t + h, y_out, k7);
    
    // Error estimate (difference between 4th and 5th order)
    for (size_t i = 0; i < n; ++i) {
        double y4 = y[i] + h * (d1*k1[i] + d2*k2[i] + d3*k3[i] + d4*k4[i] + d5*k5[i] + d6*k6[i] + d7*k7[i]);
        y_err[i] = std::abs(y_out[i] - y4);
        
        // Check for NaN or Inf
        if (!std::isfinite(y_out[i]) || !std::isfinite(y_err[i])) {
            std::cout << "NaN or Inf detected in RK45 step - solver unstable. Try smaller h_max or tighter tolerance.";
            // set large error to force step rejection
            y_err[i] = std::numeric_limits<double>::max();
            #if DEBUG_RK45
            // Debug: Print k and y_temp values
            std::cout << "RK45 Step Debug Info at t = " << t << " with h = " << h << std::endl;
            for (size_t j = 0; j < n; ++j) {
                std::cout << "k1[" << j << "] = " << k1[j] << ", k2[" << j << "] = " << k2[j]
                          << ", k3[" << j << "] = " << k3[j]
                          << ", k4[" << j << "] = " << k4[j]
                          << ", k5[" << j << "] = " << k5[j]
                          << ", k6[" << j << "] = " << k6[j]
                          << ", k7[" << j << "] = " << k7[j] << std::endl;
            }
            #endif // DEBUG_RK45
            return;
        }
    }
}

double compute_optimal_step(
    double error,
    double h_current,
    double tol,
    double safety
) {
    if (error < std::numeric_limits<double>::epsilon()) {
        return h_current * 2.0;  // Double the step if error is negligible
    }
    
    // Compute optimal step using error control formula
    double h_opt = h_current * safety * std::pow(tol / error, 0.2);
    
    return h_opt;
}

RK45Result rk45_solve_cpu(
    const ODEFunction& rhs_function,
    const std::vector<double>& y0,
    double t0,
    double tf,
    const RK45Parameters& params
) {
    RK45Result result;
    result.success = false;
    
    if (y0.empty()) {
        result.message = "Initial state vector is empty";
        return result;
    }
    
    if (tf <= t0) {
        result.message = "Final time must be greater than initial time";
        return result;
    }
    
    size_t n = y0.size();
    double t = t0;
    std::vector<double> y = y0;
    double h = std::min(params.h_max, (tf - t0) / 100.0);  // Initial step size
    
    // Store initial condition
    result.t.push_back(t);
    result.y.push_back(y);
    
    std::vector<double> y_new(n);
    std::vector<double> y_err(n);
    
    int step_iterations = 0;
    
    while (t < tf && step_iterations < params.max_iterations) {
        // Adjust step size if we would overshoot
        if (t + h > tf) {
            h = tf - t;
        }
        
        // Take a step
        rk45_step(rhs_function, t, y, h, y_new, y_err);

        #if DEBUG_RK45
            // Debug: Print current step info
            if (t == t0 || step_iterations % 1000 == 0) {
                std::cout << "t: " << t << ", h: " << h << ", y[0]: " << y[0] << std::endl;
                std::cout << "y_new[0]: " << y_new[0] << ", y_err[0]: " << y_err[0] << std::endl;
            }
        #endif // DEBUG_RK45
        
        // Compute error norm
        double error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double scale = std::max(std::abs(y[i]), std::abs(y_new[i]));
            if (scale > 0.0) {
                error += (y_err[i] / scale) * (y_err[i] / scale);
            }
        }
        error = std::sqrt(error / n);
        
        // Accept or reject step
        if (error <= params.tol || h <= params.h_min) {
            // Accept step
            t += h;
            y = y_new;
            
            result.t.push_back(t);
            result.y.push_back(y);
            
            step_iterations++;

            #if DEBUG_RK45
            // Debug: Print current step info
            if (t == t0 || step_iterations % 1000 == 0) {
                std::cout << "t: " << t << ", h: " << h << ", y[0]: " << y[0] << std::endl;
            }
            #endif // DEBUG_RK45
        }
        
        // Compute new step size
        if (error > std::numeric_limits<double>::epsilon()) {
            h = compute_optimal_step(error, h, params.tol, params.safety_factor);
            h = std::clamp(h, params.h_min, params.h_max);

            #if DEBUG_RK45
            // Debug: Print step size adjustment
            std::cout << "Adjusted step size to h: " << h << " based on error: " << error << std::endl;
            #endif // DEBUG_RK45
        }
    }
    
    result.steps = step_iterations;
    
    if (t >= tf - std::numeric_limits<double>::epsilon()) {
        result.success = true;
        result.message = "Integration completed successfully";
    } else {
        result.message = "Maximum iterations reached";
    }
    
    return result;
}

} // namespace nlo
