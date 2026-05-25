add_executable(portopt_cli main.cpp)

target_link_libraries(portopt_cli PRIVATE portopt_lib CLI11::CLI11)

target_include_directories(portopt_cli PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(portopt_cli PROPERTIES OUTPUT_NAME "po")

install(TARGETS portopt_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
