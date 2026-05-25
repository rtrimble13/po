add_executable(portopt_cli main.cpp)

target_link_libraries(portopt_cli PRIVATE portopt_lib CLI11::CLI11)

target_include_directories(portopt_cli PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(portopt_cli PROPERTIES OUTPUT_NAME "po")

install(TARGETS portopt_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Stage `po` under examples/terminal/bin so the terminal example scripts find
# it at a stable, generator-independent path. Opt-in:
#   cmake --build build --target po-examples
add_custom_target(po-examples
    DEPENDS portopt_cli
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${PROJECT_SOURCE_DIR}/examples/terminal/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:portopt_cli>
        ${PROJECT_SOURCE_DIR}/examples/terminal/bin/po
    COMMENT "Stage po binary at examples/terminal/bin/po for example scripts"
    VERBATIM
)
