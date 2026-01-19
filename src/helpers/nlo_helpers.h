#ifndef NLO_HELPERS_H
#define NLO_HELPERS_H

#define _USE_MATH_DEFINES

#include <vector>
#include <complex>
#include <cmath>

namespace nlo_helpers
{

    /**
     * @brief Convert complex to real vector representation
     *
     * @param complex_vec Vector of complex numbers
     * @return Vector of real numbers [real1, imag1, real2, imag2, ...]
     */
    std::vector<double> complex_to_real(const std::vector<std::complex<double>> &complex_vev);

    /**
     * @brief Utility function to convert real vector to complex representation
     *
     * @param real_vec Vector of real numbers [real1, imag1, real2, imag2, ...]
     * @return Vector of complex numbers
     */
    std::vector<std::complex<double>> real_to_complex(const std::vector<double> &real_vec);

}

#endif // NLO_HELPERS_H