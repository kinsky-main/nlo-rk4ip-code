# Fails when LIBRARY needs a glibc symbol version newer than FLOOR.
# Usage: cmake -DOBJDUMP=<objdump> -DLIBRARY=<libnlolib.so> -DFLOOR=2.35 -P check_glibc_floor.cmake

execute_process(
  COMMAND "${OBJDUMP}" -T "${LIBRARY}"
  OUTPUT_VARIABLE symbols
  RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "${OBJDUMP} -T ${LIBRARY} failed with ${status}")
endif()

string(REGEX MATCHALL "GLIBC_[0-9]+\\.[0-9]+(\\.[0-9]+)?" versions "${symbols}")
list(REMOVE_DUPLICATES versions)
set(newest "0")
foreach(tag IN LISTS versions)
  string(REPLACE "GLIBC_" "" version "${tag}")
  if(version VERSION_GREATER newest)
    set(newest "${version}")
  endif()
endforeach()

if(newest VERSION_GREATER FLOOR)
  string(REGEX MATCHALL "[^\n]*GLIBC_${newest}[^\n]*" offenders "${symbols}")
  string(REPLACE ";" "\n" offenders "${offenders}")
  message(FATAL_ERROR "${LIBRARY} needs glibc ${newest}, above the ${FLOOR} floor:\n${offenders}")
endif()
message(STATUS "${LIBRARY} needs glibc ${newest} at most (floor ${FLOOR})")
