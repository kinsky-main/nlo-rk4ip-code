#include <nlo_helpers.h>

namespace nlo_helpers
{

    std::vector<double> complex_to_real(const std::vector<std::complex<double>> &complex_vec)
    {
        std::vector<double> real_vec;
        real_vec.reserve(complex_vec.size() * 2);
        for (const auto &c : complex_vec)
        {
            real_vec.push_back(c.real());
            real_vec.push_back(c.imag());
        }
        return real_vec;
    }

    std::vector<std::complex<double>> real_to_complex(const std::vector<double> &real_vec)
    {
        if (real_vec.size() % 2 != 0)
        {
            throw std::invalid_argument("Input real vector size must be even.");
        }

        std::vector<std::complex<double>> complex_vec;
        complex_vec.reserve(real_vec.size() / 2);
        for (size_t i = 0; i < real_vec.size(); i += 2)
        {
            complex_vec.emplace_back(real_vec[i], real_vec[i + 1]);
        }
        return complex_vec;
    }

} // namespace nlo_helpers