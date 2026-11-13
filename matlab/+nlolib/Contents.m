% nlolib  GPU-first nonlinear optics propagation (MATLAB wrapper).
%
% Solver
%   NLolib                   - Load the nlolib library and run propagations.
%   NLolib/propagate         - Propagate a pulse or coupled (t, x, y) field.
%   NLolib/query_runtime_limits - Solver memory and record limits for a config.
%
% Logging and diagnostics
%   NLolib/set_log_level     - Runtime log level threshold.
%   NLolib/set_log_file      - Write runtime logs to a file.
%   NLolib/set_log_buffer    - Keep runtime logs in an in-memory buffer.
%   NLolib/tail_logs         - Print buffered runtime logs.
%   NLolib/set_progress_options - Progress bar behaviour.
%   NLolib/perf_profile_read - Read runtime performance counters.
%
% Operator helpers
%   translate_runtime_handle - Compile a function handle into a runtime expression.
%
% Low-level C API bridge
%   prepare_sim_config       - Build a sim_config libstruct from a MATLAB struct.
%   pack_complex_array       - Pack a complex vector for the C API.
%   unpack_records           - Unpack C output records into a MATLAB matrix.
%
% Setup
%   nlolib_setup             - Add nlolib (and its examples) to the MATLAB path.
%
% Examples are in the toolbox folder examples/matlab; the user guide is
% docs/matlab_user_guide.md.
