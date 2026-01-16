# Tests CMakeLists.txt

# Simple test executable
add_executable(test_rk45 test_rk45.cpp)
target_link_libraries(test_rk45 nlo_solver)

# Add test
add_test(NAME RK45BasicTest COMMAND test_rk45)

# Test for non-linear optics equations
add_executable(test_nlo test_nlo.cpp)
target_link_libraries(test_nlo nlo_solver)

add_test(NAME NLOEquationsTest COMMAND test_nlo)
