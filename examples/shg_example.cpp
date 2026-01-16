#include "rk45_solver.h"
#include "nlo_equations.h"
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * @brief Second Harmonic Generation (SHG) Example
 * 
 * Simulates the conversion of fundamental frequency to second harmonic
 * in a non-linear crystal.
 */

int main() {
    std::cout << "=== Second Harmonic Generation (SHG) Example ===" << std::endl;
    std::cout << "Simulating frequency doubling in a non-linear crystal" << std::endl << std::endl;
    
    // Setup non-linear optics parameters
    nlo::NLOParameters params;
    params.wavelength = 1064e-9;  // Nd:YAG wavelength (m)
    params.chi2 = 1e-12;          // Second-order susceptibility
    params.n0 = 1.5;              // Refractive index
    params.alpha = 0.01;          // Small loss (1/m)
    params.gamma = 0.0;
    
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Wavelength: " << params.wavelength * 1e9 << " nm" << std::endl;
    std::cout << "  Chi2: " << params.chi2 << " m/V" << std::endl;
    std::cout << "  Refractive index: " << params.n0 << std::endl;
    std::cout << "  Loss coefficient: " << params.alpha << " /m" << std::endl << std::endl;
    
    // Create SHG equation system
    nlo::SHGEquations shg_system(params);
    
    // Initial conditions: [A1_real, A1_imag, A2_real, A2_imag]
    // Start with fundamental field, no second harmonic
    double A1_initial = 1.0;  // Normalized amplitude
    std::vector<double> y0 = {A1_initial, 0.0, 0.0, 0.0};
    
    // Propagation distance
    double z0 = 0.0;        // Initial position (m)
    double zf = 0.01;       // Final position (1 cm)
    
    // Solver parameters
    nlo::RK45Parameters solver_params;
    solver_params.tol = 1e-9;
    solver_params.h_min = 1e-8;
    solver_params.h_max = 1e-4;
    
    // Create wrapper for the SHG equations
    auto shg_func = [&shg_system](double z, const std::vector<double>& y, std::vector<double>& dydt) {
        shg_system(z, y, dydt);
    };
    
    // Solve the system
    std::cout << "Propagating from z = " << z0 * 1000 << " mm to z = " << zf * 1000 << " mm..." << std::endl;
    auto result = nlo::rk45_solve_cpu(shg_func, y0, z0, zf, solver_params);
    
    // Print results
    std::cout << "\nStatus: " << result.message << std::endl;
    std::cout << "Number of steps: " << result.steps << std::endl;
    std::cout << "Total points: " << result.t.size() << std::endl;
    
    if (result.success) {
        std::cout << "\nConversion efficiency along propagation:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(15) << "Position (mm)" 
                  << std::setw(20) << "Fund. Intensity" 
                  << std::setw(20) << "SH Intensity"
                  << std::setw(20) << "Conversion (%)" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        for (size_t i = 0; i < result.t.size(); i += std::max(size_t(1), result.t.size() / 20)) {
            double z = result.t[i];
            
            // Extract complex amplitudes
            std::complex<double> A1(result.y[i][0], result.y[i][1]);
            std::complex<double> A2(result.y[i][2], result.y[i][3]);
            
            double I1 = nlo::compute_intensity(A1);
            double I2 = nlo::compute_intensity(A2);
            double conversion = (I2 / (I1 + I2)) * 100.0;  // Percentage
            
            std::cout << std::setw(15) << z * 1000
                      << std::setw(20) << I1
                      << std::setw(20) << I2
                      << std::setw(20) << conversion << std::endl;
        }
        
        // Final conversion efficiency
        std::complex<double> A1_final(result.y.back()[0], result.y.back()[1]);
        std::complex<double> A2_final(result.y.back()[2], result.y.back()[3]);
        double I1_final = nlo::compute_intensity(A1_final);
        double I2_final = nlo::compute_intensity(A2_final);
        double final_conversion = (I2_final / (I1_final + I2_final)) * 100.0;
        
        std::cout << "\nFinal conversion efficiency: " << std::setprecision(2) << final_conversion << "%" << std::endl;
    }
    
    return result.success ? 0 : 1;
}
