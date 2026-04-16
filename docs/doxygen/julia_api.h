/**
 * @file julia_api.h
 * @brief Docs-only Julia binding API shim for Doxygen.
 */

#pragma once

namespace julia {
namespace NLOLib {

struct NLOComplex {};
struct PropagationParams {};
struct TimeGrid {};
struct FrequencyGrid {};
struct SpatialGrid {};
struct Tensor3DDesc {};

/**
 * @ingroup julia_binding
 * @brief Runtime operator expression/configuration struct.
 */
struct RuntimeOperatorParams {};

/**
 * @ingroup julia_binding
 * @brief Simulation-only propagation configuration.
 */
struct SimulationConfig {};

/**
 * @ingroup julia_binding
 * @brief Physics/operator configuration.
 */
struct PhysicsConfig {};

struct VulkanBackendConfig {};
struct ExecutionOptions {};

/**
 * @ingroup julia_binding
 * @brief Runtime-derived solver/resource limits.
 */
struct RuntimeLimits {};

/**
 * @ingroup julia_binding
 * @brief Snapshot storage configuration.
 */
struct StorageOptions {};

/**
 * @ingroup julia_binding
 * @brief Snapshot storage result summary.
 */
struct StorageResult {};

/**
 * @ingroup julia_binding
 * @brief Accepted-step progress event payload.
 */
struct StepEvent {};

/**
 * @ingroup julia_binding
 * @brief Prepared simulation/physics config pair.
 */
struct PreparedSimConfig {};

/**
 * @ingroup julia_binding
 * @brief Prepared native-backed value wrapper.
 */
struct PreparedValue {};

/**
 * @ingroup julia_binding
 * @brief High-level runtime operator specification.
 */
struct RuntimeOperators {};

/**
 * @ingroup julia_binding
 * @brief Input pulse and optional tensor-grid metadata.
 */
struct PulseSpec {};

/**
 * @ingroup julia_binding
 * @brief Linear or nonlinear operator specification.
 */
struct OperatorSpec {};

/**
 * @ingroup julia_binding
 * @brief Per-event propagation progress metadata.
 */
struct ProgressInfo {};

/**
 * @ingroup julia_binding
 * @brief High-level propagation result.
 */
struct PropagationResult {};

/**
 * @ingroup julia_binding
 * @brief Convenience result returned by propagate.
 */
struct PropagateResult {};

/**
 * @ingroup julia_binding
 * @brief Raised when a progress callback aborts propagation.
 *
 * Native Julia type name: \c NLOLib.PropagationAbortedError.
 */
class PropagationAbortedError {};

/**
 * @ingroup julia_binding
 * @brief Low-allocation Julia client facade.
 *
 * Native Julia type name: \c NLOLib.NLolib.
 */
class NLolib {
public:
    NLolib();
};

/**
 * @ingroup julia_binding
 * @brief Load the shared nlolib library.
 */
void load();

/**
 * @ingroup julia_binding
 * @brief Return the currently loaded library path.
 */
const char* loaded_library_path();

/**
 * @ingroup julia_binding
 * @brief Return whether SQLite-backed storage is available.
 */
int storage_is_available();

/**
 * @ingroup julia_binding
 * @brief Construct a prepared physics config.
 */
PhysicsConfig physics_config();

/**
 * @ingroup julia_binding
 * @brief Construct prepared storage options.
 */
StorageOptions storage_options();

/**
 * @ingroup julia_binding
 * @brief Build default execution options.
 */
ExecutionOptions default_execution_options();

/**
 * @ingroup julia_binding
 * @brief Build default storage options.
 */
StorageOptions default_storage_options();

/**
 * @ingroup julia_binding
 * @brief Query runtime-derived solver limits.
 */
RuntimeLimits query_runtime_limits();

/**
 * @ingroup julia_binding
 * @brief Propagate into caller-owned output buffers.
 *
 * Native Julia function name: \c NLOLib.propagate!.
 */
PropagateResult propagate_bang();

/**
 * @ingroup julia_binding
 * @brief High-level propagation entrypoint.
 */
PropagateResult propagate();

/**
 * @ingroup julia_binding
 * @brief Wrap native records as a Julia matrix view.
 */
void wrap_records();

/**
 * @ingroup julia_binding
 * @brief Return the final propagated record view.
 */
void final_record();

/**
 * @ingroup julia_binding
 * @brief Interpret a record as a tensor-grid view.
 */
void tensor_record_view();

/**
 * @ingroup julia_binding
 * @brief Prepare split simulation and physics configs.
 */
PreparedSimConfig prepare_sim_config();

/**
 * @ingroup julia_binding
 * @brief Configure a file sink for runtime logs.
 */
void set_log_file();

/**
 * @ingroup julia_binding
 * @brief Configure an in-memory runtime log buffer.
 */
void set_log_buffer();

/**
 * @ingroup julia_binding
 * @brief Clear buffered runtime logs.
 */
void clear_log_buffer();

/**
 * @ingroup julia_binding
 * @brief Read buffered runtime logs.
 */
const char* read_log_buffer();

/**
 * @ingroup julia_binding
 * @brief Set the global runtime log level.
 */
void set_log_level();

/**
 * @ingroup julia_binding
 * @brief Configure runtime progress output options.
 */
void set_progress_options();

/**
 * @ingroup julia_binding
 * @brief Configure the runtime progress output stream.
 */
void set_progress_stream();

} // namespace NLOLib
} // namespace julia
