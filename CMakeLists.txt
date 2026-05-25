cmake_minimum_required(VERSION 3.20)
project(portopt
    VERSION 1.0.0
    DESCRIPTION "Portfolio Optimization Library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(PORTOPT_BUILD_PYTHON   "Build Python bindings"    ON)
option(PORTOPT_BUILD_CLI      "Build CLI tool"           ON)
option(PORTOPT_BUILD_TESTS    "Build test suite"         ON)
option(PORTOPT_BUILD_EXAMPLES "Build example programs"   ON)
# Off by default: dependencies fetched via FetchContent aren't in any export
# set, so the namespaced install of portopt_lib fails at configure time.
# Enable only when consuming portopt via find_package downstream.
option(PORTOPT_INSTALL        "Generate install/export targets" OFF)

# Position-independent code for shared library / pybind11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Dependencies via FetchContent ────────────────────────────────────────────
include(FetchContent)

# Eigen (linear algebra)
FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Eigen)

# nlohmann/json
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# toml++ (header-only TOML parser)
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tomlplusplus)

# spdlog
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spdlog)

# CLI11 (command-line parsing)
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(CLI11)

# ── Core library ─────────────────────────────────────────────────────────────
add_subdirectory(src)

# ── Python bindings ───────────────────────────────────────────────────────────
if(PORTOPT_BUILD_PYTHON)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    add_subdirectory(python)
endif()

# ── CLI ───────────────────────────────────────────────────────────────────────
if(PORTOPT_BUILD_CLI)
    add_subdirectory(cli)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(PORTOPT_BUILD_TESTS)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
if(PORTOPT_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS portopt_lib
        EXPORT  portoptTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/portopt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT portoptTargets
        FILE      portoptTargets.cmake
        NAMESPACE portopt::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/portopt
    )
endif()
