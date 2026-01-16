#include "rk45_solver.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

/**
 * @brief Simple Harmonic Oscillator Example
 * 
 * Solves the differential equation:
 * d²x/dt² = -omega²x
 * 
 * Converted to first-order system:
 * dx/dt = v
 * dv/dt = -omega²x
 */

int main() {
    std::cout << "=== Harmonic Oscillator Example ===" << std::endl;
    std::cout << "Solving: d²x/dt² = -omega²x with omega = 1.0" << std::endl;
    std::cout << "Initial conditions: x(0) = 1.0, v(0) = 0.0" << std::endl << std::endl;
    
    // System parameters
    const double omega = 1.0;  // Angular frequency
    
    // Define the ODE system
    auto harmonic_oscillator = [omega](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        // y[0] = x (position)
        // y[1] = v (velocity)
        dydt[0] = y[1];                    // dx/dt = v
        dydt[1] = -omega * omega * y[0];   // dv/dt = -omega²x
    };
    
    // Initial conditions
    std::vector<double> y0 = {1.0, 0.0, 1.0, 0.0};  // x(0) = 1, v(0) = 0
    
    // Time range
    double t0 = 0.0;
    double tf = 20.0;  // 10 seconds
    
    // Solver parameters
    nlo::RK45Parameters params;
    params.tol = 1e-4;
    params.h_min = 1e-6;
    params.h_max = 0.1;
    
    // Solve the system
    std::cout << "Solving with RK45 adaptive step solver..." << std::endl;
    auto result = nlo::rk45_solve_cpu(harmonic_oscillator, y0, t0, tf, params);
    
    // Print results
    std::cout << "\nStatus: " << result.message << std::endl;
    std::cout << "Number of steps: " << result.steps << std::endl;
    std::cout << "Total time points: " << result.t.size() << std::endl;
    
    if (result.success) {
        std::cout << "\nSample of solution (every 10th point):" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(12) << "Time" 
                  << std::setw(15) << "Position (x)" 
                  << std::setw(15) << "Velocity (v)"
                  << std::setw(15) << "Analytic x"
                  << std::setw(15) << "Error" << std::endl;
        std::cout << std::string(72, '-') << std::endl;
        
        for (size_t i = 0; i < result.t.size(); i += std::max(size_t(1), result.t.size() / 20)) {
            double t = result.t[i];
            double x_numeric = result.y[i][0];
            double v_numeric = result.y[i][1];
            double x_analytic = std::cos(omega * t);  // Analytic solution
            double error = std::abs(x_numeric - x_analytic);
            
            std::cout << std::setw(12) << t 
                      << std::setw(15) << x_numeric
                      << std::setw(15) << v_numeric
                      << std::setw(15) << x_analytic
                      << std::setw(15) << error << std::endl;
        }
        
        // Compute final error
        double final_t = result.t.back();
        double final_x = result.y.back()[0];
        double final_x_analytic = std::cos(omega * final_t);
        double final_error = std::abs(final_x - final_x_analytic);
        
        std::cout << "\nFinal error: " << std::scientific << final_error << std::endl;
        
        // Check for directory to output data and create data dir if not exists
        std::filesystem::path output_dir("data");
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        // Output data to csv for external plotting
        std::ofstream outfile("data/harmonic_oscillator_solution.csv");
        outfile << "Time,Position,Velocity,Analytic_Position,Error\n";
        for (size_t i = 0; i < result.t.size(); ++i)
        {
            double t = result.t[i];
            double x_numeric = result.y[i][0];
            double v_numeric = result.y[i][1];
            double x_analytic = std::cos(omega * t);
            double error = std::abs(x_numeric - x_analytic);
            outfile << t << "," << x_numeric << "," << v_numeric << "," 
                    << x_analytic << "," << error << "\n";
        }
    }
    
    return result.success ? 0 : 1;
}
