try:
    from nlolib_cffi import load, ffi
except ModuleNotFoundError:
    print("test_python_bindings: cffi not installed; skipping.")
    raise SystemExit(0)

lib = load()  # uses NLOLIB_LIBRARY env
print("test_python_bindings: loaded nlolib CFFI bindings.")
cfg = ffi.new("sim_config*")
cfg.nonlinear.gamma = 1.0
cfg.dispersion.num_dispersion_terms = 0
cfg.propagation.starting_step_size = 0.01
cfg.propagation.max_step_size = 0.1
cfg.propagation.min_step_size = 0.001
cfg.propagation.propagation_distance = 0.0
cfg.time.pulse_period = 1.0
cfg.time.delta_time = 0.001
print("test_python_bindings: configured sim_config scalar fields.")

n = 128
inp = ffi.new("nlo_complex[]", n)
out = ffi.new("nlo_complex[]", n)

cfg.dispersion.alpha = 0.0

freq = ffi.new("nlo_complex[]", n)
cfg.frequency.frequency_grid = freq
print("test_python_bindings: assigned frequency grid buffer.")

assert cfg.dispersion.num_dispersion_terms == 0
assert cfg.frequency.frequency_grid != ffi.NULL
print("test_python_bindings: verified struct field access.")

status = lib.nlolib_propagate(cfg, n, inp, out)
assert int(status) == 0
print("test_python_bindings: nlolib_propagate returned expected status.")
