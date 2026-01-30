<!-- General Idea -->

# Data Handling

This module will handle the storage and retrieval of simulation data. The issues to address include:
- Memory limitations when running large simulations, having to store data in memory limits efficiency of the solver as it removes valuable memory resources from the simulation.
- As the simulation iterates (ideally in GPU device shared memory), any output data will be transferred to the host memory.
- The target performance limitation should be GPU device memory limiting grid sizes, not host memory. GPU bandwidth on modern cards is very high, so as long as only steps over a reasonable time frame or resolution are stored in host memory, the transfer time should be negligible compared to the simulation time.

# Proposed Solution

- Package in SQLite into the library as a static library.
- Create a data handling module that manages the SQLite database.
- The module will provide functions to initialize a database, create tables for simulation parameters and results, and insert data as the simulation progresses.
- The module will be exposed as a public API to the user so they can open the connection and define what data to store. 