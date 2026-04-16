/**
 * @file matlab_api.h
 * @brief Docs-only MATLAB binding API shim for Doxygen.
 */

#pragma once

namespace matlab {
namespace nlolib {

struct StructValue {};
struct StepHistory {};
struct StorageResult {};
struct RuntimeLimits {};

/**
 * @ingroup matlab_binding
 * @brief High-level MATLAB wrapper around the nlolib shared library.
 *
 * Native MATLAB class name: \c %nlolib.NLolib.
 */
class NLolib {
public:
    NLolib(const char* libraryPath = "");
    StructValue propagate(const StructValue& primary, const StructValue& varargin = StructValue());
    int storage_is_available() const;
    void set_log_file(const char* path, int append = 0);
    void set_log_buffer(unsigned long long capacityBytes = 262144u);
    void clear_log_buffer();
    const char* read_log_buffer(int consume = 1, unsigned long long maxBytes = 262144u) const;
    const char* tail_logs(int consume = 1, unsigned long long maxBytes = 262144u) const;
    void set_log_level(int level);
    void set_progress_options(int enabled = 1, int milestonePercent = 5, int emitOnStepAdjust = 0);
    void set_progress_stream(int streamMode);
    RuntimeLimits query_runtime_limits(const StructValue& config = StructValue(),
                                       const StructValue& execOptions = StructValue()) const;
};

/**
 * @ingroup matlab_binding
 * @brief Translate a MATLAB function handle into a runtime expression plus constants.
 *
 * Native MATLAB function name: \c %nlolib.translate_runtime_handle.
 */
void translate_runtime_handle();

/**
 * @ingroup matlab_binding
 * @brief Prepare split simulation and physics structs for the native entry points.
 *
 * Native MATLAB function name: \c %nlolib.prepare_sim_config.
 */
void prepare_sim_config();

/**
 * @ingroup matlab_binding
 * @brief Pack a MATLAB complex array into the native buffer layout.
 *
 * Native MATLAB function name: \c %nlolib.pack_complex_array.
 */
void pack_complex_array();

/**
 * @ingroup matlab_binding
 * @brief Unpack propagated records into MATLAB arrays.
 *
 * Native MATLAB function name: \c %nlolib.unpack_records.
 */
void unpack_records();

} // namespace nlolib
} // namespace matlab
