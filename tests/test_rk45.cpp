#include "rk45_solver.h"
#include <iostream>
#include <cassert>

#define _USE_MATH_DEFINES
#include <cmath>

/**
 * @brief Basic tests for RK45 solver
 */

// Test 1: Exponential decay dy/dt = -y, y(0) = 1
bool test_exponential_decay() {
    std::cout << "Test 1: Exponential decay" << std::endl;
    
    auto exponential = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -y[0];
    };
    
    std::vector<double> y0 = {1.0};
    double t0 = 0.0;
    double tf = 5.0;
    
    nlo::RK45Parameters params;
    params.tol = 1e-10;
    
    auto result = nlo::rk45_solve_cpu(exponential, y0, t0, tf, params);
    
    if (!result.success) {
        std::cout << "  FAILED: " << result.message << std::endl;
        return false;
    }
    
    // Check final value against analytic solution
    double y_numeric = result.y.back()[0];
    double y_analytic = std::exp(-tf);
    double error = std::abs(y_numeric - y_analytic);
    
    std::cout << "  Numeric: " << y_numeric << std::endl;
    std::cout << "  Analytic: " << y_analytic << std::endl;
    std::cout << "  Error: " << error << std::endl;
    
    if (error > 1e-8) {
        std::cout << "  FAILED: Error too large" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 2: Linear system
bool test_linear_system() {
    std::cout << "\nTest 2: Linear system" << std::endl;
    
    auto linear = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[0] + 2*y[1];
        dydt[1] = 3*y[0] + 2*y[1];
    };
    
    std::vector<double> y0 = {1.0, 0.0};
    double t0 = 0.0;
    double tf = 1.0;
    
    nlo::RK45Parameters params;
    params.tol = 1e-9;
    
    auto result = nlo::rk45_solve_cpu(linear, y0, t0, tf, params);
    
    if (!result.success) {
        std::cout << "  FAILED: " << result.message << std::endl;
        return false;
    }
    
    std::cout << "  Steps: " << result.steps << std::endl;
    std::cout << "  Final y[0]: " << result.y.back()[0] << std::endl;
    std::cout << "  Final y[1]: " << result.y.back()[1] << std::endl;
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 3: Harmonic oscillator
bool test_harmonic_oscillator() {
    std::cout << "\nTest 3: Harmonic oscillator" << std::endl;
    
    const double omega = 1.0;
    auto oscillator = [omega](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];
        dydt[1] = -omega * omega * y[0];
    };
    
    std::vector<double> y0 = {1.0, 0.0};  // x=1, v=0
    double t0 = 0.0;
    double tf = 2.0 * M_PI;  // One period
    
    nlo::RK45Parameters params;
    params.tol = 1e-10;
    
    auto result = nlo::rk45_solve_cpu(oscillator, y0, t0, tf, params);
    
    if (!result.success) {
        std::cout << "  FAILED: " << result.message << std::endl;
        return false;
    }
    
    // After one period, should return to initial state
    double x_final = result.y.back()[0];
    double v_final = result.y.back()[1];
    double error_x = std::abs(x_final - 1.0);
    double error_v = std::abs(v_final - 0.0);
    
    std::cout << "  Final x: " << x_final << " (expected 1.0, error: " << error_x << ")" << std::endl;
    std::cout << "  Final v: " << v_final << " (expected 0.0, error: " << error_v << ")" << std::endl;
    
    if (error_x > 1e-8 || error_v > 1e-8) {
        std::cout << "  FAILED: Error too large" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 4: Adaptive stepping
bool test_adaptive_stepping() {
    std::cout << "\nTest 4: Adaptive stepping" << std::endl;
    
    // Stiff problem that requires adaptive stepping
    auto stiff = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -100.0 * y[0];
    };
    
    std::vector<double> y0 = {1.0};
    double t0 = 0.0;
    double tf = 1.0;
    
    nlo::RK45Parameters params;
    params.tol = 1e-6;
    params.h_min = 1e-8;
    params.h_max = 0.1;
    
    auto result = nlo::rk45_solve_cpu(stiff, y0, t0, tf, params);
    
    if (!result.success) {
        std::cout << "  FAILED: " << result.message << std::endl;
        return false;
    }
    
    std::cout << "  Steps taken: " << result.steps << std::endl;
    std::cout << "  Points generated: " << result.t.size() << std::endl;
    
    // Verify that adaptive stepping worked (more steps due to stiffness)
    if (result.steps < 10) {
        std::cout << "  FAILED: Too few steps for stiff problem" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== RK45 Solver Tests ===" << std::endl << std::endl;
    
    int passed = 0;
    int total = 4;
    
    if (test_exponential_decay()) passed++;
    if (test_linear_system()) passed++;
    if (test_harmonic_oscillator()) passed++;
    if (test_adaptive_stepping()) passed++;
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}
