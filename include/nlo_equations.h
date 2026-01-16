#ifndef NLO_EQUATIONS_H
#define NLO_EQUATIONS_H

#define _USE_MATH_DEFINES

#include <vector>
#include <complex>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

/**
 * @brief Non-linear optics equations and utility functions
 * 
 * This header provides common non-linear optics equations that can be solved
 * using the RK45 solver, including:
 * - Second Harmonic Generation (SHG)
 * - Self-Phase Modulation (SPM)
 * - Four-Wave Mixing (FWM)
 */

namespace nlo {

/**
 * @brief Parameters for non-linear optics simulations
 */
struct NLOParameters {
    double n0 = 1.5;              // Linear refractive index
    double n2 = 2.6e-20;          // Non-linear refractive index (m^2/W)
    double beta2 = -1e-26;        // Group velocity dispersion (s^2/m)
    double gamma = 1.0;           // Non-linearity coefficient (1/W/m)
    double alpha = 0.00004;       // Loss coefficient (1/m)
    double wavelength = 1550e-9;  // Wavelength (m)
    double chi2 = 0.0;            // Second-order susceptibility
    double f_R = 0.18;        // Raman fraction
    double tau1 = 12.2e-15;     // Raman time constant 1 (s)
    double tau2 = 32e-15;       // Raman time constant 2 (s)
    double omega0 = 0;
    std::vector<double> beta; // Higher-order dispersion coefficients
};

/**
 * @brief Second Harmonic Generation (SHG) equations
 * 
 * Models the conversion of fundamental frequency to second harmonic.
 * State vector: [A1_real, A1_imag, A2_real, A2_imag]
 * where A1 is fundamental and A2 is second harmonic
 */
class SHGEquations {
public:
    explicit SHGEquations(const NLOParameters& params);
    
    /**
     * @brief Compute derivatives for SHG equations
     * 
     * @param z Propagation distance
     * @param y State vector [A1_real, A1_imag, A2_real, A2_imag]
     * @param dydt Output derivatives
     */
    void operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const;
    
private:
    NLOParameters params_;
    double delta_k_;  // Phase mismatch
};

/**
 * @brief Self-Phase Modulation (SPM) with dispersion
 * 
 * Models pulse propagation in a non-linear medium with SPM and GVD.
 * State vector: [A_real, A_imag]
 * where A is the complex amplitude
 */
class SPMEquations {
public:
    explicit SPMEquations(const NLOParameters& params);
    
    void operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const;
    
private:
    NLOParameters params_;
};

/**
 * @brief Non-linear Schrödinger Equation (NLSE)
 * 
 * General form of NLSE including dispersion and non-linearity.
 * This is the fundamental equation for pulse propagation in fibers.
 */
class NLSEEquations {
public:
    explicit NLSEEquations(const NLOParameters& params);
    
    void operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const;
    
    // Set the temporal grid for frequency domain operations
    void set_temporal_grid(const std::vector<double>& t);
    
private:
    NLOParameters params_;
    std::vector<double> t_grid_;
};

/**
 * @brief Generalized NLSE (GNLSE) with higher-order effects
 * 
 * Extends NLSE to include higher-order dispersion and Raman effects.
 * State vector: [A_real, A_imag]
 * 
 * In this case considering the simplified form:
 * 
 * dif / (dif z) A(z) = -alpha/2 A(z) - i/2 beta_2 dif^2 / (dif t^2) A(z) + i gamma |A(z)|^2 A(z)
 * 
 */
class GNLSEEquations {
public:
    explicit GNLSEEquations(const NLOParameters& params);
    ~GNLSEEquations();
    
    /**
     * @brief Non-linear operator in time domain
     * 
     * @param A_time Complex amplitude in time domain
     * @param dA_dz_time Output derivative in time domain
     */
    void linear_operator(
        const std::vector<std::complex<double>>& A_time,
        std::vector<std::complex<double>>& dA_dz_time
    ) const;
    
    /**
     * @brief Linear operator in frequency domain
     * 
     * @param A_freq Complex amplitude in frequency domain
     * @param dA_dz_freq Output derivative in frequency domain
     */
    void non_linear_operator(
        const std::vector<std::complex<double>>& A_freq,
        std::vector<std::complex<double>>& dA_dz_freq
    ) const;

    /**
     * @brief Compute derivatives for GNLSE equations
     * 
     * @param z Propagation distance
     * @param y State vector [A_real, A_imag]
     * @param dydt Output derivatives
     */
    void operator()(
        double z,
        const std::vector<double>& y,
        std::vector<double>& dydt
    ) const;

    void set_temporal_grid(const std::vector<double>& t);

private:
    NLOParameters params_; //!< Non-linear optics parameters
    std::vector<double> t_grid_; //!< Temporal grid
    std::vector<double> omega_grid_; //!< Frequency grid
    std::vector<double> beta_of_omega_; //!< Dispersion polynomial coefficients
    std::vector<std::complex<double>> L_omega_; //!< Linear operator in frequency domain

    std::vector<std::complex<double>> h_R_t_; //!< Raman response function in time domain
    std::vector<std::complex<double>> H_R_omega_; //!< FFT(H_R_t_) for convolution in frequency domain

    std::size_t N_{0}; //!< Number of grid points
    double dt_{0.0};   //!< Temporal step size

    cufftHandle fft_plan_{0}; //!< CUFFT plan for forward FFT
    cufftDoubleComplex* d_fft_buffer_{nullptr}; //!< Device buffer for FFT operations

    void init_fft();
    void destroy_fft();

    void fft_forward(
        const std::vector<std::complex<double>>& in,
        std::vector<std::complex<double>>& out
    ) const;

    void fft_inverse(
        const std::vector<std::complex<double>>& in,
        std::vector<std::complex<double>>& out
    ) const;

    inline void cuda_check(cudaError_t err, const char* msg) const {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " +
                                    cudaGetErrorString(err));
        }
    }

    inline void cufft_check(cufftResult res, const char* msg) const {
        if (res != CUFFT_SUCCESS) {
            throw std::runtime_error(std::string(msg) + ": cuFFT error code " +
                                    std::to_string(res));
        }
    }
};

/**
 * @brief Utility function to convert complex to real vector representation
 * 
 * @param complex_vec Vector of complex numbers
 * @return Vector of real numbers [real1, imag1, real2, imag2, ...]
 */
std::vector<double> complex_to_real(const std::vector<std::complex<double>>& complex_vec);

/**
 * @brief Utility function to convert real vector to complex representation
 * 
 * @param real_vec Vector of real numbers [real1, imag1, real2, imag2, ...]
 * @return Vector of complex numbers
 */
std::vector<std::complex<double>> real_to_complex(const std::vector<double>& real_vec);

/**
 * @brief Compute intensity from complex amplitude
 * 
 * @param amplitude Complex amplitude
 * @return Intensity (|amplitude|^2)
 */
double compute_intensity(const std::complex<double>& amplitude);

/**
 * @brief Compute phase from complex amplitude
 * 
 * @param amplitude Complex amplitude
 * @return Phase in radians
 */
double compute_phase(const std::complex<double>& amplitude);

} // namespace nlo

#endif // NLO_EQUATIONS_H
