pybind11_add_module(_portopt bindings.cpp)

target_link_libraries(_portopt PRIVATE portopt_lib)

target_include_directories(_portopt PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Install into the Python package directory
install(TARGETS _portopt
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python/portopt
)
