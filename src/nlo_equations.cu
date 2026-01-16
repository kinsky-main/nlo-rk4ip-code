// cSpell: disable
#define _USE_MATH_DEFINES
#include "nlo_equations.h"
#include <cmath>
#include <cuda_runtime.h>

namespace nlo {

// Constants
const double PI = 3.14159265358979323846;
const double C_LIGHT = 299792458.0;  // Speed of light in m/s

// Helper functions
std::vector<double> complex_to_real(const std::vector<std::complex<double>>& complex_vec) {
    std::vector<double> real_vec(complex_vec.size() * 2);
    for (size_t i = 0; i < complex_vec.size(); ++i) {
        real_vec[2*i] = complex_vec[i].real();
        real_vec[2*i + 1] = complex_vec[i].imag();
    }
    return real_vec;
}

std::vector<std::complex<double>> real_to_complex(const std::vector<double>& real_vec) {
    std::vector<std::complex<double>> complex_vec(real_vec.size() / 2);
    for (size_t i = 0; i < complex_vec.size(); ++i) {
        complex_vec[i] = std::complex<double>(real_vec[2*i], real_vec[2*i + 1]);
    }
    return complex_vec;
}

double compute_intensity(const std::complex<double>& amplitude) {
    return std::norm(amplitude);
}

double compute_phase(const std::complex<double>& amplitude) {
    return std::arg(amplitude);
}

// SHG Equations Implementation
SHGEquations::SHGEquations(const NLOParameters& params) 
    : params_(params), delta_k_(0.0) {
    // Compute phase mismatch
    // For perfect phase matching, delta_k = 0
    // Otherwise: delta_k = k2 - 2*k1
    double k1 = 2.0 * PI * params_.n0 / params_.wavelength;
    double k2 = 2.0 * PI * params_.n0 / (params_.wavelength / 2.0);
    delta_k_ = k2 - 2.0 * k1;
}

void SHGEquations::operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const {
    // State vector: [A1_real, A1_imag, A2_real, A2_imag]
    // A1: fundamental frequency
    // A2: second harmonic
    
    if (y.size() != 4 || dydt.size() != 4) {
        throw std::invalid_argument("SHG equations require state vector of size 4");
    }
    
    std::complex<double> A1(y[0], y[1]);
    std::complex<double> A2(y[2], y[3]);
    
    // Coupled equations for SHG:
    // dA1/dz = -i * chi2 * A2 * conj(A1) * exp(i*delta_k*z)
    // dA2/dz = -i * chi2 * A1^2 * exp(-i*delta_k*z)
    
    std::complex<double> phase_factor(std::cos(delta_k_ * z), std::sin(delta_k_ * z));
    std::complex<double> i_unit(0.0, 1.0);
    
    std::complex<double> dA1dz = -i_unit * params_.chi2 * A2 * std::conj(A1) * phase_factor;
    std::complex<double> dA2dz = -i_unit * params_.chi2 * A1 * A1 * std::conj(phase_factor);
    
    // Include loss
    dA1dz -= params_.alpha * A1 / 2.0;
    dA2dz -= params_.alpha * A2 / 2.0;
    
    dydt[0] = dA1dz.real();
    dydt[1] = dA1dz.imag();
    dydt[2] = dA2dz.real();
    dydt[3] = dA2dz.imag();
}

// SPM Equations Implementation
SPMEquations::SPMEquations(const NLOParameters& params) 
    : params_(params) {}

void SPMEquations::operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const {
    // State vector: [A_real, A_imag]
    
    if (y.size() != 2 || dydt.size() != 2) {
        throw std::invalid_argument("SPM equations require state vector of size 2");
    }
    
    std::complex<double> A(y[0], y[1]);
    std::complex<double> i_unit(0.0, 1.0);
    
    // SPM equation: dA/dz = i * gamma * |A|^2 * A - alpha/2 * A
    double intensity = std::norm(A);
    
    std::complex<double> dAdz = i_unit * params_.gamma * intensity * A - (params_.alpha / 2.0) * A;
    
    dydt[0] = dAdz.real();
    dydt[1] = dAdz.imag();
}

// NLSE Equations Implementation
NLSEEquations::NLSEEquations(const NLOParameters& params) 
    : params_(params) {}

void NLSEEquations::operator()(double z, const std::vector<double>& y, std::vector<double>& dydt) const {
    // Generalized NLSE with dispersion and non-linearity
    // State vector: [A_real, A_imag] for each time point
    
    if (y.size() != 2 || dydt.size() != 2) {
        throw std::invalid_argument("NLSE equations require state vector of size 2");
    }
    
    std::complex<double> A(y[0], y[1]);
    std::complex<double> i_unit(0.0, 1.0);
    
    // Simplified NLSE (without full dispersion operator):
    // dA/dz = i * gamma * |A|^2 * A - alpha/2 * A
    // Full dispersion would require FFT, which will be handled by the interaction picture method.
    double intensity = std::norm(A);
    std::complex<double> dAdz = i_unit * params_.gamma * intensity * A - (params_.alpha / 2.0) * A;
    
    dydt[0] = dAdz.real();
    dydt[1] = dAdz.imag();
}

void NLSEEquations::set_temporal_grid(const std::vector<double>& t) {
    t_grid_ = t;
}

GNLSEEquations::GNLSEEquations(const NLOParameters& params) 
    : params_(params) {}

GNLSEEquations::~GNLSEEquations() {
    destroy_fft();
}

void GNLSEEquations::destroy_fft() {
    if (fft_plan_ != 0) {
        cufftDestroy(fft_plan_);
        fft_plan_ = 0;
    }
    if (d_fft_buffer_ != nullptr) {
        cudaFree(d_fft_buffer_);
        d_fft_buffer_ = nullptr;
    }
}

void GNLSEEquations::init_fft() {
    destroy_fft();  // in case we are re‐initialising with a new N_

    if (N_ == 0) {
        throw std::runtime_error("N_ not set before init_fft");
    }

    // Allocate device buffer (in‐place transform)
    cuda_check(cudaMalloc(&d_fft_buffer_,
                          N_ * sizeof(cufftDoubleComplex)),
               "cudaMalloc for d_fft_buffer");

    // Create 1D double-complex plan
    // Z2Z = complex<double> -> complex<double>
    cufft_check(cufftPlan1d(&fft_plan_, static_cast<int>(N_),
                            CUFFT_Z2Z, 1),
                "cufftPlan1d");
}

void GNLSEEquations::fft_forward(
    const std::vector<std::complex<double>>& in,
    std::vector<std::complex<double>>& out
) const
{
    if (in.size() != N_) {
        throw std::runtime_error("fft_forward: size mismatch");
    }
    if (fft_plan_ == 0 || d_fft_buffer_ == nullptr) {
        throw std::runtime_error("fft_forward: FFT plan not initialised");
    }

    // Pack input into temporary host array of cufftDoubleComplex
    std::vector<cufftDoubleComplex> host(N_);
    for (std::size_t i = 0; i < N_; ++i) {
        host[i].x = in[i].real();
        host[i].y = in[i].imag();
    }

    // Host -> device
    cuda_check(cudaMemcpy(d_fft_buffer_, host.data(),
                          N_ * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D in fft_forward");

    // Execute FFT in place
    cufft_check(cufftExecZ2Z(fft_plan_,
                             d_fft_buffer_,
                             d_fft_buffer_,
                             CUFFT_FORWARD),
                "cufftExecZ2Z forward");

    // Device -> host
    cuda_check(cudaMemcpy(host.data(), d_fft_buffer_,
                          N_ * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H in fft_forward");

    // Unpack into std::complex
    out.resize(N_);
    for (std::size_t i = 0; i < N_; ++i) {
        out[i] = std::complex<double>(host[i].x, host[i].y);
    }
}

void GNLSEEquations::fft_inverse(
    const std::vector<std::complex<double>>& in,
    std::vector<std::complex<double>>& out
) const
{
    if (in.size() != N_) {
        throw std::runtime_error("fft_inverse: size mismatch");
    }
    if (fft_plan_ == 0 || d_fft_buffer_ == nullptr) {
        throw std::runtime_error("fft_inverse: FFT plan not initialised");
    }

    std::vector<cufftDoubleComplex> host(N_);
    for (std::size_t i = 0; i < N_; ++i) {
        host[i].x = in[i].real();
        host[i].y = in[i].imag();
    }

    cuda_check(cudaMemcpy(d_fft_buffer_, host.data(),
                          N_ * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D in fft_inverse");

    cufft_check(cufftExecZ2Z(fft_plan_,
                             d_fft_buffer_,
                             d_fft_buffer_,
                             CUFFT_INVERSE),
                "cufftExecZ2Z inverse");

    cuda_check(cudaMemcpy(host.data(), d_fft_buffer_,
                          N_ * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H in fft_inverse");

    out.resize(N_);
    const double scale = 1.0 / static_cast<double>(N_);
    for (std::size_t i = 0; i < N_; ++i) {
        out[i] = std::complex<double>(host[i].x * scale,
                                      host[i].y * scale);
    }
}

void GNLSEEquations::linear_operator(
    const std::vector<std::complex<double>>& A_freq,
    std::vector<std::complex<double>>& dA_dz_freq
) const
{
    const std::size_t N = A_freq.size();
    if (N != N_) {
        throw std::runtime_error("Size mismatch in linear_operator");
    }

    dA_dz_freq.resize(N);
    for (std::size_t k = 0; k < N; ++k) {
        dA_dz_freq[k] = L_omega_[k] * A_freq[k];
    }
}


void GNLSEEquations::non_linear_operator(
    const std::vector<std::complex<double>>& A_time,
    std::vector<std::complex<double>>& dA_dz_time
) const {
    const std::size_t N = A_time.size();
    if (N != N_) {
        throw std::runtime_error("Size mismatch in non_linear_operator");
    }

    dA_dz_time.resize(N);

    // Intensity |A|^2
    std::vector<double> I(N);
    for (std::size_t i = 0; i < N; ++i) {
        I[i] = std::norm(A_time[i]);
    }

    // Raman convolution: R(t) = (h_R * I)(t)
    std::vector<std::complex<double>> I_c(N), R_c(N);
    for (std::size_t i = 0; i < N; ++i) {
        I_c[i] = std::complex<double>(I[i], 0.0);
    }

    std::vector<std::complex<double>> I_omega(N), R_omega(N);
    fft_forward(I_c, I_omega);

    for (std::size_t k = 0; k < N; ++k) {
        R_omega[k] = I_omega[k] * H_R_omega_[k];
    }

    fft_inverse(R_omega, R_c); // back to time: R_c ~ convolution

    // Real Raman term (imaginary should be ~0)
    std::vector<double> R(N);
    for (std::size_t i = 0; i < N; ++i) {
        R[i] = R_c[i].real();
    }

    const double gamma = params_.gamma;
    const double f_R   = params_.f_R;

    for (std::size_t i = 0; i < N; ++i) {
        double I_inst = I[i];
        double R_val  = R[i];

        // Nonlinear part: iγ [ (1-f_R)|A|^2 + f_R R(t) ] A
        double nl_coeff = (1.0 - f_R) * I_inst + f_R * R_val;
        std::complex<double> rhs =
            std::complex<double>(0.0, gamma * nl_coeff) * A_time[i];

        dA_dz_time[i] = rhs;
    }
}
void GNLSEEquations::set_temporal_grid(const std::vector<double>& t)
{
    t_grid_ = t;
    N_ = t_grid_.size();
    if (N_ < 2) {
        throw std::runtime_error("Temporal grid must have at least 2 points");
    }

    dt_ = t_grid_[1] - t_grid_[0];
    
    // Initialize FFT with the new grid size
    init_fft();
    
    // (You may want to assert |t[i+1]-t[i]-dt_| is small if grid is not perfectly uniform)
    omega_grid_.resize(N_);

    // Frequency spacing: df = 1 / (N * dt)
    const double df = 1.0 / (static_cast<double>(N_) * dt_);
    const double two_pi = 2.0 * M_PI;

    for (std::size_t k = 0; k < N_; ++k) {
        // standard symmetric frequency layout:
        double f;
        if (k <= N_ / 2) {
            f = static_cast<double>(k) * df;
        } else {
            f = static_cast<double>(k - N_) * df;
        }
        omega_grid_[k] = two_pi * f;
    }

    // --- Build beta(omega) polynomial around omega0 ---
    beta_of_omega_.resize(N_);

    // params_.beta[m] corresponds to beta_{m+2}
    for (std::size_t k = 0; k < N_; ++k) {
        double d_omega = omega_grid_[k] - params_.omega0;
        double beta_val = 0.0;

        double pow_dw = d_omega * d_omega; // (Delta omega)^2, for m=2
        for (std::size_t j = 0; j < params_.beta.size(); ++j) {
            std::size_t m = j + 2; // order
            // beta_m / m! * (Delta omega)^m
            double fact = 1.0;
            for (std::size_t q = 2; q <= m; ++q) fact *= static_cast<double>(q);
            beta_val += params_.beta[j] * pow_dw / fact;
            pow_dw *= d_omega; // next power
        }
        beta_of_omega_[k] = beta_val;
    }

    // Linear operator eigenvalues in frequency domain:
    // L(omega) = -alpha/2 + i beta(omega)
    L_omega_.resize(N_);
    for (std::size_t k = 0; k < N_; ++k) {
        L_omega_[k] = std::complex<double>(
            -0.5 * params_.alpha,
            beta_of_omega_[k]
        );
    }

    // Raman response h_R(t) in time domain
    h_R_t_.resize(N_);
    double norm = 0.0;

    const double tau1 = params_.tau1;
    const double tau2 = params_.tau2;

    for (std::size_t i = 0; i < N_; ++i) {
        double t_i = t_grid_[i];
        double h = 0.0;
        if (t_i >= 0.0) {
            // standard silica Raman response:
            // h_R(t) = ((τ1^2 + τ2^2)/(τ1 τ2^2)) exp(-t/τ2) sin(t/τ1)
            double prefactor = (tau1*tau1 + tau2*tau2) / (tau1 * tau2 * tau2);
            h = prefactor * std::exp(-t_i / tau2) * std::sin(t_i / tau1);
        }
        h_R_t_[i] = h;
        norm += h * dt_;
    }

    // Normalize so integral h_R(t) dt = 1
    if (norm > 0.0) {
        for (auto& v : h_R_t_) {
            v /= norm;
        }
    }

    // Precompute FFT(h_R_t) for fast convolution
    std::vector<std::complex<double>> h_R_c(N_);
    for (std::size_t i = 0; i < N_; ++i) {
        h_R_c[i] = h_R_t_[i];
    }
    H_R_omega_.resize(N_);
    fft_forward(h_R_c, H_R_omega_);
}

void GNLSEEquations::operator()(
    double z,
    const std::vector<double>& y,
    std::vector<double>& dydt
) const {
    // Convert real representation to complex (this is A_I in interaction picture)
    std::vector<std::complex<double>> A_I_time = real_to_complex(y);
    
    if (A_I_time.size() != N_) {
        throw std::runtime_error("Size mismatch in GNLSE operator()");
    }
    
    // Step 1: Transform to frequency domain
    std::vector<std::complex<double>> A_I_freq;
    fft_forward(A_I_time, A_I_freq);
    
    // Step 2: Apply exp(L*z) to get A(z,t) from A_I(z,t)
    // A_freq = exp(L*z) * A_I_freq
    std::vector<std::complex<double>> A_freq(N_);
    for (std::size_t k = 0; k < N_; ++k) {
        A_freq[k] = std::exp(L_omega_[k] * z) * A_I_freq[k];
    }
    
    // Step 3: Transform back to time domain to get A(z,t)
    std::vector<std::complex<double>> A_time;
    fft_inverse(A_freq, A_time);
    
    // Step 4: Compute nonlinear operator N(A) in time domain
    std::vector<std::complex<double>> N_time;
    non_linear_operator(A_time, N_time);
    
    // Step 5: Transform N(A) to frequency domain
    std::vector<std::complex<double>> N_freq;
    fft_forward(N_time, N_freq);
    
    // Step 6: Apply exp(-L*z) to get back to interaction picture
    // dA_I/dz = exp(-L*z) * N(A)
    std::vector<std::complex<double>> dA_I_dz_freq(N_);
    for (std::size_t k = 0; k < N_; ++k) {
        dA_I_dz_freq[k] = std::exp(-L_omega_[k] * z) * N_freq[k];
    }
    
    // Step 7: Transform back to time domain
    std::vector<std::complex<double>> dA_I_dz_time;
    fft_inverse(dA_I_dz_freq, dA_I_dz_time);
    
    // Convert back to real representation
    dydt = complex_to_real(dA_I_dz_time);
}


} // namespace nlo
