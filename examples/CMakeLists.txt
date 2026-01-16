# Examples CMakeLists.txt

# Simple harmonic oscillator example
add_executable(harmonic_oscillator harmonic_oscillator.cpp)
target_link_libraries(harmonic_oscillator nlo_solver)

# Second Harmonic Generation example
add_executable(shg_example shg_example.cpp)
target_link_libraries(shg_example nlo_solver)

# Self-Phase Modulation example
add_executable(spm_example spm_example.cpp)
target_link_libraries(spm_example nlo_solver SQLite::SQLite3)

# Install examples
install(TARGETS harmonic_oscillator shg_example spm_example
    RUNTIME DESTINATION bin/examples
)
