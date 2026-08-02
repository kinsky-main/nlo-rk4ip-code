# check_matlab_binding_types.cmake
#
# Verifies that every C type name the MATLAB wrapper passes to libstruct() /
# libpointer() is actually defined by the flattened FFI header
# src/nlolib_matlab.h.
#
# loadlibrary() resolves these names at run time from strings, so a renamed C
# type silently breaks the binding with no compile-time signal. This check is
# the static guard for that failure mode: it caught nothing at the time it was
# written only because the `nlo_complex` -> `complex` rename had already been
# reverted.
#
# Runs in CMake script mode; requires no MATLAB installation.
#
#   cmake -DNLOLIB_SOURCE_DIR=<repo> -P cmake/check_matlab_binding_types.cmake

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED NLOLIB_SOURCE_DIR)
  message(FATAL_ERROR "NLOLIB_SOURCE_DIR must be defined")
endif()

set(_header "${NLOLIB_SOURCE_DIR}/src/nlolib_matlab.h")
if(NOT EXISTS "${_header}")
  message(FATAL_ERROR "MATLAB FFI header not found: ${_header}")
endif()

file(READ "${_header}" _header_text)

# ---------------------------------------------------------------------------
# Collect type names the header defines.
# ---------------------------------------------------------------------------
set(_known_types "")

# NOTE: CMake splits string(REGEX MATCHALL) output into a list on ';', so a
# pattern ending in ';' loses that terminator once iterated. The extraction
# patterns below therefore stop before the semicolon.

# `} name;`  -- struct/enum typedefs (the dominant form in this header).
string(REGEX MATCHALL "\n}[ \t]*[A-Za-z_][A-Za-z0-9_]*" _closers "${_header_text}")
foreach(_closer IN LISTS _closers)
  string(REGEX REPLACE "^\n}[ \t]*" "" _name "${_closer}")
  if(_name)
    list(APPEND _known_types "${_name}")
  endif()
endforeach()

# `typedef existing new;` -- plain aliases such as physics_config.
string(REGEX MATCHALL "typedef[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]+[A-Za-z_][A-Za-z0-9_]*"
       _aliases "${_header_text}")
foreach(_alias IN LISTS _aliases)
  string(REGEX REPLACE "^typedef[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]+" "" _name "${_alias}")
  if(_name)
    list(APPEND _known_types "${_name}")
  endif()
endforeach()

# `typedef void *name;` -- the opaque Vulkan handle stubs.
string(REGEX MATCHALL "typedef[ \t]+void[ \t]*\\*[ \t]*[A-Za-z_][A-Za-z0-9_]*"
       _voidptrs "${_header_text}")
foreach(_voidptr IN LISTS _voidptrs)
  string(REGEX REPLACE "^typedef[ \t]+void[ \t]*\\*[ \t]*" "" _name "${_voidptr}")
  if(_name)
    list(APPEND _known_types "${_name}")
  endif()
endforeach()

list(REMOVE_DUPLICATES _known_types)
list(LENGTH _known_types _known_count)
if(_known_count LESS 10)
  message(FATAL_ERROR
    "Parsed only ${_known_count} type names from ${_header}; the extraction "
    "regexes are probably stale. Refusing to pass vacuously.")
endif()

# MATLAB builtin pointer types that never appear in the header.
set(_builtin_types
    voidPtr  int8Ptr    uint8Ptr   int16Ptr  uint16Ptr
    int32Ptr uint32Ptr  int64Ptr   uint64Ptr
    singlePtr doublePtr cstring    stringPtrPtr voidPtrPtr)

# ---------------------------------------------------------------------------
# Scan the wrapper sources.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _m_files "${NLOLIB_SOURCE_DIR}/matlab/*.m")
if(NOT _m_files)
  message(FATAL_ERROR "No MATLAB sources found under ${NLOLIB_SOURCE_DIR}/matlab")
endif()

set(_errors "")
set(_checked 0)

foreach(_m_file IN LISTS _m_files)
  file(STRINGS "${_m_file}" _lines)
  set(_lineno 0)
  foreach(_line IN LISTS _lines)
    math(EXPR _lineno "${_lineno} + 1")

    # Skip comment-only lines so documentation examples do not trip the check.
    string(REGEX MATCH "^[ \t]*%" _is_comment "${_line}")
    if(_is_comment)
      continue()
    endif()

    # libstruct('name') / libpointer('namePtr') / setdatatype(p,'namePtr',..)
    string(REGEX MATCHALL "(libstruct|libpointer|setdatatype)\\([^)]*'([A-Za-z_][A-Za-z0-9_]*)'"
           _uses "${_line}")
    foreach(_use IN LISTS _uses)
      string(REGEX REPLACE ".*'([A-Za-z_][A-Za-z0-9_]*)'$" "\\1" _type "${_use}")

      list(FIND _builtin_types "${_type}" _builtin_idx)
      if(NOT _builtin_idx EQUAL -1)
        continue()
      endif()

      math(EXPR _checked "${_checked} + 1")

      # Strip a trailing Ptr to get the underlying C type name.
      string(REGEX REPLACE "Ptr$" "" _base "${_type}")

      list(FIND _known_types "${_base}" _idx)
      if(_idx EQUAL -1)
        file(RELATIVE_PATH _rel "${NLOLIB_SOURCE_DIR}" "${_m_file}")
        list(APPEND _errors
             "  ${_rel}:${_lineno}: '${_type}' -> no type '${_base}' in src/nlolib_matlab.h")
      endif()
    endforeach()
  endforeach()
endforeach()

if(_errors)
  string(REPLACE ";" "\n" _error_text "${_errors}")
  message(FATAL_ERROR
    "MATLAB binding references C types the FFI header does not define:\n"
    "${_error_text}\n\n"
    "loadlibrary() resolves these names at run time, so this breaks every\n"
    "call that touches the type. Fix the .m file or add the type to\n"
    "src/nlolib_matlab.h.")
endif()

message(STATUS
  "MATLAB binding type check: ${_checked} type references OK "
  "against ${_known_count} header types.")
