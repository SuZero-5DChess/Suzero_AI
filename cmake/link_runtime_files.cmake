# Stage runtime DLLs next to a C++ executable.
#
# A hardlink costs no additional disk space when the build tree and the
# runtime source live on the same volume (the normal case), and it keeps the
# executable runnable straight from its output directory. COPY_ON_ERROR falls
# back to a regular copy when linking is impossible (different volume, FAT,
# or a filesystem without hardlink support).
#
# Inputs (pass with -D):
#   SUZERO_RUNTIME_SOURCE_DIR  directory holding the source DLLs
#   SUZERO_RUNTIME_TARGET_DIR  output directory of the target
#   SUZERO_RUNTIME_FILES       semicolon-separated DLL names

foreach(_name IN LISTS SUZERO_RUNTIME_FILES)
  set(_source "${SUZERO_RUNTIME_SOURCE_DIR}/${_name}")
  if(NOT EXISTS "${_source}")
    message(STATUS "Runtime file not found, skipping: ${_source}")
    continue()
  endif()

  set(_target "${SUZERO_RUNTIME_TARGET_DIR}/${_name}")
  file(CREATE_LINK "${_source}" "${_target}" COPY_ON_ERROR RESULT _result)
  if(NOT _result STREQUAL "0")
    message(WARNING "Could not stage ${_name}: ${_result}")
  endif()
endforeach()
