if(NOT DEFINED TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE is required.")
endif()

set(_csv "${TEST_SOURCE_DIR}/benchmarks/results/test_tensor_dry_run.csv")
set(_storage "${TEST_SOURCE_DIR}/benchmarks/results/test_tensor_dry_run_storage")

execute_process(
  COMMAND "${TEST_EXECUTABLE}"
          --scenario=tensor3d_scaling
          --dry-run
          --tensor-scales=8,16,32,64
          --planner-host-bytes=40000000
          --planner-gpu-bytes=5000000
          --csv=${_csv}
          --storage-dir=${_storage}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
)

if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Dry-run benchmark failed.\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
endif()

string(FIND "${_stdout}" "gpu_fit" _has_gpu_fit)
string(FIND "${_stdout}" "host_fit_only" _has_host_fit_only)
string(FIND "${_stdout}" "output_spill" _has_output_spill)

if(_has_gpu_fit LESS 0 OR _has_host_fit_only LESS 0 OR _has_output_spill LESS 0)
  message(FATAL_ERROR
    "Dry-run output did not contain all expected tensor regions.\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
endif()
