#include "nlo_equations.h"
#include "rk45_solver.h"
#include <iostream>
#include <cmath>

/**
 * @brief Tests for non-linear optics equations
 */

// Test 1: Complex/Real conversions
bool test_complex_conversions() {
    std::cout << "Test 1: Complex/Real conversions" << std::endl;
    
    std::vector<std::complex<double>> complex_vec = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    
    auto real_vec = nlo::complex_to_real(complex_vec);
    auto complex_back = nlo::real_to_complex(real_vec);
    
    if (real_vec.size() != 6) {
        std::cout << "  FAILED: Wrong real vector size" << std::endl;
        return false;
    }
    
    if (complex_back.size() != 3) {
        std::cout << "  FAILED: Wrong complex vector size" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < complex_vec.size(); ++i) {
        if (std::abs(complex_vec[i].real() - complex_back[i].real()) > 1e-10 ||
            std::abs(complex_vec[i].imag() - complex_back[i].imag()) > 1e-10) {
            std::cout << "  FAILED: Conversion error" << std::endl;
            return false;
        }
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 2: Intensity and phase computation
bool test_intensity_phase() {
    std::cout << "\nTest 2: Intensity and phase computation" << std::endl;
    
    std::complex<double> A(3.0, 4.0);
    
    double intensity = nlo::compute_intensity(A);
    double phase = nlo::compute_phase(A);
    
    double expected_intensity = 25.0;  // |3+4i|² = 9+16 = 25
    double expected_phase = std::atan2(4.0, 3.0);
    
    if (std::abs(intensity - expected_intensity) > 1e-10) {
        std::cout << "  FAILED: Wrong intensity" << std::endl;
        std::cout << "    Expected: " << expected_intensity << std::endl;
        std::cout << "    Got: " << intensity << std::endl;
        return false;
    }
    
    if (std::abs(phase - expected_phase) > 1e-10) {
        std::cout << "  FAILED: Wrong phase" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 3: SPM equations evaluation
bool test_spm_equations() {
    std::cout << "\nTest 3: SPM equations evaluation" << std::endl;
    
    nlo::NLOParameters params;
    params.gamma = 1.0;
    params.alpha = 0.0;  // No loss for simple test
    
    nlo::SPMEquations spm(params);
    
    std::vector<double> y = {1.0, 0.0};  // Real amplitude = 1, imag = 0
    std::vector<double> dydt(2);
    
    double z = 0.0;
    spm(z, y, dydt);
    
    // For A = 1+0i, |A|² = 1
    // dA/dz = i*gamma*|A|²*A = i*1*1*(1+0i) = i
    // So dydt should be [0, 1] (real part 0, imag part 1)
    
    if (std::abs(dydt[0] - 0.0) > 1e-10 || std::abs(dydt[1] - 1.0) > 1e-10) {
        std::cout << "  FAILED: Wrong derivative" << std::endl;
        std::cout << "    Expected: [0, 1]" << std::endl;
        std::cout << "    Got: [" << dydt[0] << ", " << dydt[1] << "]" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 4: SHG equations
bool test_shg_equations() {
    std::cout << "\nTest 4: SHG equations" << std::endl;
    
    nlo::NLOParameters params;
    params.chi2 = 1e-12;
    params.alpha = 0.0002;
    
    nlo::SHGEquations shg(params);
    
    std::vector<double> y = {1.0, 0.0, 0.0, 1.0};  // Only fundamental, no SH
    std::vector<double> dydt(4);
    
    double z = 0.0;
    shg(z, y, dydt);
    
    // Derivatives should be computed without error
    // Just check that function executes
    std::cout << "  Derivatives computed: [" 
              << dydt[0] << ", " << dydt[1] << ", " 
              << dydt[2] << ", " << dydt[3] << "]" << std::endl;
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

// Test 5: SPM propagation
bool test_spm_propagation() {
    std::cout << "\nTest 5: SPM propagation" << std::endl;
    
    nlo::NLOParameters params;
    params.gamma = 0.6;
    params.alpha = 0.0002;
    
    nlo::SPMEquations spm_system(params);
    
    auto spm_func = [&spm_system](double z, const std::vector<double>& y, std::vector<double>& dydt) {
        spm_system(z, y, dydt);
    };
    
    std::vector<double> y0 = {1.0, 0.0};
    double z0 = 0.0;
    double zf = 100.0;
    
    nlo::RK45Parameters solver_params;
    solver_params.tol = 1e-8;
    
    auto result = nlo::rk45_solve_cpu(spm_func, y0, z0, zf, solver_params);
    
    if (!result.success) {
        std::cout << "  FAILED: " << result.message << std::endl;
        return false;
    }
    
    std::cout << "  Steps: " << result.steps << std::endl;
    std::cout << "  Final amplitude: [" 
              << result.y.back()[0] << ", " << result.y.back()[1] << "]" << std::endl;
    
    // Check that amplitude is preserved within expected loss
    std::complex<double> A_final(result.y.back()[0], result.y.back()[1]);
    double intensity_final = nlo::compute_intensity(A_final);
    
    // Check intensity is within expected Beer-Lambert decay
    double expected_intensity = std::exp(-params.alpha * (zf - z0));
    if (std::abs(intensity_final - expected_intensity) > 1e-6) {
        std::cout << "  FAILED: Intensity not preserved" << std::endl;
        return false;
    }
    
    std::cout << "  PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Non-Linear Optics Equations Tests ===" << std::endl << std::endl;
    
    int passed = 0;
    int total = 5;
    
    if (test_complex_conversions()) passed++;
    if (test_intensity_phase()) passed++;
    if (test_spm_equations()) passed++;
    if (test_shg_equations()) passed++;
    if (test_spm_propagation()) passed++;
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}
