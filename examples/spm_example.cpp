#include "rk45_solver.h"
#include "nlo_equations.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sqlite3.h>

#define _USE_MATH_DEFINES
#include <cmath>

/**
 * @brief Self-Phase Modulation (SPM) Example
 * 
 * Simulates pulse propagation with SPM in an optical fiber.
 * SPM causes spectral broadening due to intensity-dependent refractive index.
 */

int main(int argc, char* argv[]) {
    std::cout << "=== Self-Phase Modulation (SPM) Example ===" << std::endl;
    std::cout << "Simulating Gaussian pulse propagation in optical fiber" << std::endl << std::endl;
    
    // Default parameters
    double wavelength = 1550e-9;  // Telecom wavelength (m)
    double gamma = 0.01;          // Non-linearity coefficient (1/W/m)
    double beta2 = -2.0e-10;      // GVD parameter (s^2/m)
    double alpha_dB = 0.2;        // Fiber loss (dB/km)
    double n0 = 1.45;             // Refractive index
    double P0 = 1000.0;           // Peak power (W)
    double T0 = 50e-12;           // Pulse width (s)
    double zf = 1.0;             // Fiber length (m)
    size_t N_t = 512;             // Number of time points
    
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Error: Missing value for parameter " << argv[i] << std::endl;
            return 1;
        }
        std::string param = argv[i];
        double value = std::atof(argv[i + 1]);
        
        if (param == "--wavelength" || param == "-w") {
            wavelength = value * 1e-9;  // Input in nm
        } else if (param == "--gamma" || param == "-g") {
            gamma = value;
        } else if (param == "--beta2" || param == "-b") {
            beta2 = value * 1e-27;  // Input in ps^2/km
        } else if (param == "--alpha" || param == "-a") {
            alpha_dB = value;
        } else if (param == "--n0" || param == "-n") {
            n0 = value;
        } else if (param == "--power" || param == "-p") {
            P0 = value;
        } else if (param == "--pulse-width" || param == "-t") {
            T0 = value * 1e-12;  // Input in ps
        } else if (param == "--length" || param == "-l") {
            zf = value;
        } else if (param == "--time-points" || param == "-N") {
            N_t = static_cast<size_t>(value);
        } else if (param == "--help" || param == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --wavelength, -w <nm>      Wavelength (default: 1550 nm)\n";
            std::cout << "  --gamma, -g <1/W/m>        Nonlinearity coefficient (default: 0.01)\n";
            std::cout << "  --beta2, -b <ps^2/km>      GVD parameter (default: -2.0)\n";
            std::cout << "  --alpha, -a <dB/km>        Fiber loss (default: 0.2)\n";
            std::cout << "  --n0, -n <value>           Refractive index (default: 1.45)\n";
            std::cout << "  --power, -p <W>            Peak power (default: 1000 W)\n";
            std::cout << "  --pulse-width, -t <ps>     Pulse width T0 (default: 50 ps)\n";
            std::cout << "  --length, -l <m>           Fiber length (default: 10 m)\n";
            std::cout << "  --time-points, -N <int>    Number of time points (default: 512)\n";
            std::cout << "  --help, -h                 Show this help message\n";
            return 0;
        } else {
            std::cerr << "Error: Unknown parameter " << param << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
    }
    
    // Setup fiber parameters
    nlo::NLOParameters params;
    params.wavelength = wavelength;
    params.gamma = gamma;
    params.beta2 = beta2;
    params.alpha = alpha_dB * std::log(10) / 10000;  // Convert dB/km to 1/m
    params.n0 = n0;
    
    std::cout << "Fiber Parameters:" << std::endl;
    std::cout << "  Wavelength: " << params.wavelength * 1e9 << " nm" << std::endl;
    std::cout << "  Gamma: " << params.gamma << " /W/m" << std::endl;
    std::cout << "  Beta2 (GVD): " << params.beta2 * 1e27 << " ps^2/km" << std::endl;
    std::cout << "  Loss: " << alpha_dB << " dB/km" << std::endl;
    std::cout << "  Refractive index: " << params.n0 << std::endl << std::endl;
    
    // Temporal grid parameters for Gaussian pulse
    double t_window = 8.0 * T0;   // Time window: 10x pulse width
    
    // Create temporal grid centered at t=0
    std::vector<double> t_grid(N_t);
    double dt = t_window / N_t;
    for (size_t i = 0; i < N_t; ++i) {
        t_grid[i] = -t_window/2.0 + i * dt;
    }
    
    // Initial Gaussian pulse: A(t,z=0) = sqrt(P0) * exp(-t^2/(2*T0^2))
    std::vector<std::complex<double>> A_initial(N_t);
    for (size_t i = 0; i < N_t; ++i) {
        double t = t_grid[i];
        A_initial[i] = std::sqrt(P0) * std::exp(-t*t / (2.0*T0*T0));
    }
    
    // Convert to real representation [A1_real, A1_imag, A2_real, A2_imag, ...]
    std::vector<double> y0 = nlo::complex_to_real(A_initial);
    
    // Create GNLSE equation system (handles SPM + dispersion properly)
    nlo::GNLSEEquations gnlse_system(params);
    gnlse_system.set_temporal_grid(t_grid);
    
    // Propagation distance
    double z0 = 0.0;        // Initial position (m)
    
    // Solver parameters
    nlo::RK45Parameters solver_params;
    solver_params.tol = 1e-8;        // Tighter tolerance
    solver_params.h_min = 1e-8;      // Much smaller minimum step
    solver_params.h_max = 1e-4;      // Much smaller maximum step (was 0.01)
    solver_params.max_iterations = 500000;  // More iterations allowed
    
    std::cout << "Pulse Parameters:" << std::endl;
    std::cout << "  Peak power: " << P0 << " W" << std::endl;
    std::cout << "  Pulse width (T0): " << T0 * 1e12 << " ps" << std::endl;
    std::cout << "  Time window: " << t_window * 1e12 << " ps" << std::endl;
    std::cout << "  Number of time points: " << N_t << std::endl;
    std::cout << "  Nonlinear length: " << 1.0/(params.gamma * P0) << " m" << std::endl;
    std::cout << "  Dispersion length: " << T0*T0/std::abs(params.beta2) << " m" << std::endl << std::endl;
    
    // Create wrapper for GNLSE equations
    auto gnlse_func = [&gnlse_system](double z, const std::vector<double>& y, std::vector<double>& dydt) {
        gnlse_system(z, y, dydt);
    };
    
    // Solve the system
    std::cout << "Propagating from z = " << z0 << " m to z = " << zf << " m..." << std::endl;
    auto result = nlo::rk45_solve_cpu(gnlse_func, y0, z0, zf, solver_params);
    
    // Print results
    std::cout << "\nStatus: " << result.message << std::endl;
    std::cout << "Number of steps: " << result.steps << std::endl;
    std::cout << "Total z-points: " << result.t.size() << std::endl;
    
    if (result.success) {
        std::cout << "\nPulse evolution along fiber (central time point):" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(15) << "Position (m)" 
                  << std::setw(18) << "Peak Amplitude" 
                  << std::setw(18) << "Peak Power (W)"
                  << std::setw(18) << "Phase (rad)"
                  << std::setw(18) << "Power (dB)" << std::endl;
        std::cout << std::string(87, '-') << std::endl;
        
        double initial_peak_power = 0.0;
        size_t center_idx = N_t / 2;  // Center of temporal grid
        
        for (size_t i = 0; i < result.t.size(); i += std::max(size_t(1), result.t.size() / 20)) {
            double z = result.t[i];
            
            // Extract complex amplitude at center time point
            std::complex<double> A(result.y[i][2*center_idx], result.y[i][2*center_idx + 1]);
            
            double amplitude = std::abs(A);
            double peak_power = amplitude * amplitude;  // |A|^2
            double phase = nlo::compute_phase(A);
            
            if (i == 0) {
                initial_peak_power = peak_power;
            }
            
            // Power in dB relative to input
            double power_dB = 10 * std::log10(peak_power / initial_peak_power);
            
            std::cout << std::setw(15) << z
                      << std::setw(18) << amplitude
                      << std::setw(18) << peak_power
                      << std::setw(18) << phase
                      << std::setw(18) << power_dB << std::endl;
        }
        
        // Final statistics
        std::complex<double> A_final(result.y.back()[2*center_idx], result.y.back()[2*center_idx + 1]);
        double final_peak_power = std::abs(A_final) * std::abs(A_final);
        double final_phase = nlo::compute_phase(A_final);
        double total_loss_dB = 10 * std::log10(final_peak_power / initial_peak_power);
        
        std::cout << "\nFinal Statistics:" << std::endl;
        std::cout << "  Total loss: " << std::setprecision(2) << total_loss_dB << " dB" << std::endl;
        std::cout << "  Peak phase shift: " << std::setprecision(4) << final_phase << " rad" << std::endl;
        std::cout << "  Phase shift / pi: " << final_phase / M_PI << " pi" << std::endl;

        // Output to data directory and check for its existence
        std::filesystem::path output_dir("data");
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        // Create db file with sqlite3 if it doesn't exist and then write results
        std::filesystem::path db_path = output_dir / "spm_results.db";
        sqlite3* db;
        if (sqlite3_open(db_path.string().c_str(), &db) != SQLITE_OK
            ) {
            std::cerr << "Error creating/opening database: " << sqlite3_errmsg(db) << std::endl;
            return 1;
        }
        const char* create_table_sql = 
            "CREATE TABLE IF NOT EXISTS spm_results ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "z_position REAL,"
            "time_point REAL,"
            "amplitude_real REAL,"
            "amplitude_imag REAL,"
            "amplitude_abs REAL,"
            "power REAL,"
            "phase REAL"
            ");";
        char* errmsg = nullptr;
        if (sqlite3_exec(db, create_table_sql, nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::cerr << "Error creating table: " << errmsg << std::endl;
            sqlite3_free(errmsg);
            sqlite3_close(db);
            return 1;
        }
        // Begin transaction for batch insert (much faster)
        if (sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::cerr << "Error beginning transaction: " << errmsg << std::endl;
            sqlite3_free(errmsg);
            sqlite3_close(db);
            return 1;
        }
        
        const char* insert_sql = 
            "INSERT INTO spm_results (z_position, time_point, amplitude_real, amplitude_imag, amplitude_abs, power, phase) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, insert_sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Error preparing insert statement: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_close(db);
            return 1;
        }
        
        // Batch size for optimal performance (can adjust based on memory)
        const size_t batch_size = 5000;
        size_t total_inserted = 0;
        
        // Store full temporal profile at each z position
        for (size_t z_idx = 0; z_idx < result.t.size(); ++z_idx) {
            double z = result.t[z_idx];
            
            // Loop over all time points
            for (size_t t_idx = 0; t_idx < N_t; ++t_idx) {
                double t = t_grid[t_idx];
                std::complex<double> A(result.y[z_idx][2*t_idx], result.y[z_idx][2*t_idx + 1]);
                double amplitude_abs = std::abs(A);
                double power = amplitude_abs * amplitude_abs;
                double phase = nlo::compute_phase(A);
                
                sqlite3_bind_double(stmt, 1, z);
                sqlite3_bind_double(stmt, 2, t);
                sqlite3_bind_double(stmt, 3, A.real());
                sqlite3_bind_double(stmt, 4, A.imag());
                sqlite3_bind_double(stmt, 5, amplitude_abs);
                sqlite3_bind_double(stmt, 6, power);
                sqlite3_bind_double(stmt, 7, phase);
                
                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    std::cerr << "Error inserting data: " << sqlite3_errmsg(db) << std::endl;
                    sqlite3_finalize(stmt);
                    sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
                    sqlite3_close(db);
                    return 1;
                }
                sqlite3_reset(stmt);
                total_inserted++;
                
                // Commit batch and start new transaction every batch_size records
                if (total_inserted % batch_size == 0 && total_inserted < result.t.size() * N_t) {
                    sqlite3_finalize(stmt);
                    if (sqlite3_exec(db, "COMMIT;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
                        std::cerr << "Error committing batch: " << errmsg << std::endl;
                        sqlite3_free(errmsg);
                        sqlite3_close(db);
                        return 1;
                    }
                    if (sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
                        std::cerr << "Error beginning new transaction: " << errmsg << std::endl;
                        sqlite3_free(errmsg);
                        sqlite3_close(db);
                        return 1;
                    }
                    if (sqlite3_prepare_v2(db, insert_sql, -1, &stmt, nullptr) != SQLITE_OK) {
                        std::cerr << "Error re-preparing insert statement: " << sqlite3_errmsg(db) << std::endl;
                        sqlite3_close(db);
                        return 1;
                    }
                }
            }
        }
        sqlite3_finalize(stmt);
        
        // Commit final transaction
        if (sqlite3_exec(db, "COMMIT;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::cerr << "Error committing final transaction: " << errmsg << std::endl;
            sqlite3_free(errmsg);
            sqlite3_close(db);
            return 1;
        }
        
        std::cout << "\nSuccessfully inserted " << total_inserted << " records into database." << std::endl;

        sqlite3_close(db);
    }
    
    return result.success ? 0 : 1;
}
