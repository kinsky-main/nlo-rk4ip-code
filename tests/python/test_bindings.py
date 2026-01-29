from nlolib_cffi import load, ffi

lib = load()  # uses NLOLIB_LIBRARY env
cfg = ffi.new("sim_config*")
cfg.nonlinear.gamma = 1.0
cfg.dispersion.num_dispersion_terms = 0
cfg.propagation.starting_step_size = 0.1
cfg.propagation.max_step_size = 0.1
cfg.propagation.min_step_size = 0.1
cfg.propagation.propagation_distance = 1.0
cfg.time.pulse_period = 1.0
cfg.time.delta_time = 0.01

n = 8
inp = ffi.new("nlo_complex[]", n)
out = ffi.new("nlo_complex[]", n)

status = lib.nlolib_propogate(cfg, n, inp, out)
assert int(status) in (0, 3)  # 3 = NOT_IMPLEMENTED in current stub
