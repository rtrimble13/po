add_executable(portopt_tests
    test_main.cpp
    test_qp_solver.cpp
    test_mvo.cpp
    test_mvo_extensions.cpp
    test_black_litterman.cpp
    test_estimation.cpp
    test_io.cpp
    test_integration.cpp
    test_portfolios.cpp
)

target_link_libraries(portopt_tests
    PRIVATE
        portopt_lib
        Catch2::Catch2WithMain
)

target_include_directories(portopt_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Make test data directory accessible as a compile-time constant
target_compile_definitions(portopt_tests PRIVATE
    PORTOPT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

include(CTest)
include(Catch)
catch_discover_tests(portopt_tests)
