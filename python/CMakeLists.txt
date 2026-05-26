# ── pybind11 extension ────────────────────────────────────────────────────────
pybind11_add_module(_portopt bindings.cpp)

target_link_libraries(_portopt PRIVATE portopt_lib)

target_include_directories(_portopt PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Stage the compiled module next to the pure-Python package after each build,
# so `pip install -e python/` produces a working `import portopt` without any
# PYTHONPATH gymnastics. The staged copy is gitignored.
set(_portopt_staged "${PROJECT_SOURCE_DIR}/python/portopt/$<TARGET_FILE_NAME:_portopt>")
add_custom_command(TARGET _portopt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:_portopt> "${_portopt_staged}"
    COMMENT "Staging _portopt extension into python/portopt/"
)

# Optional install rule (only when PORTOPT_INSTALL is on). Lands under
# <prefix>/python/portopt — useful for downstream packaging, but the normal
# Python install path is `pip install` into a venv (see python-venv target).
if(PORTOPT_INSTALL)
    install(TARGETS _portopt
        DESTINATION ${CMAKE_INSTALL_PREFIX}/python/portopt
    )
endif()

# ── python-venv target ────────────────────────────────────────────────────────
# Creates .venv/ at the repo root and installs the portopt package (with all
# optional deps) into it in editable mode. Depends on _portopt so the .so is
# staged before pip runs. Idempotent: rerunning is fast if nothing changed.
if(NOT DEFINED PORTOPT_VENV_DIR)
    set(PORTOPT_VENV_DIR "${PROJECT_SOURCE_DIR}/.venv" CACHE PATH
        "Location of the Python virtual environment created by `python-venv`.")
endif()
if(NOT DEFINED PORTOPT_VENV_EXTRAS)
    set(PORTOPT_VENV_EXTRAS "all" CACHE STRING
        "Extras to install from python/pyproject.toml (comma-separated, or 'all').")
endif()
if(NOT DEFINED PORTOPT_PYTHON)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(PORTOPT_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
        "Python interpreter used to bootstrap the venv.")
endif()

if(WIN32)
    set(_venv_python "${PORTOPT_VENV_DIR}/Scripts/python.exe")
else()
    set(_venv_python "${PORTOPT_VENV_DIR}/bin/python")
endif()

add_custom_command(
    OUTPUT  "${_venv_python}"
    COMMAND "${PORTOPT_PYTHON}" -m venv "${PORTOPT_VENV_DIR}"
    COMMAND "${_venv_python}" -m pip install --upgrade pip
    COMMENT "Creating Python venv at ${PORTOPT_VENV_DIR}"
    VERBATIM
)

# Stamp file makes the install step idempotent — pip is only rerun when the
# pyproject or the compiled extension changes.
set(_venv_stamp "${CMAKE_CURRENT_BINARY_DIR}/.python-venv.stamp")
add_custom_command(
    OUTPUT  "${_venv_stamp}"
    DEPENDS "${_venv_python}"
            "${PROJECT_SOURCE_DIR}/python/pyproject.toml"
            _portopt
    COMMAND "${_venv_python}" -m pip install -e
            "${PROJECT_SOURCE_DIR}/python[${PORTOPT_VENV_EXTRAS}]"
    COMMAND ${CMAKE_COMMAND} -E touch "${_venv_stamp}"
    COMMENT "Installing portopt[${PORTOPT_VENV_EXTRAS}] into ${PORTOPT_VENV_DIR}"
    VERBATIM
)

add_custom_target(python-venv DEPENDS "${_venv_stamp}")
