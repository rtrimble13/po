/**
 * @file test_main.cpp
 * @brief Catch2 test runner entry point and shared test fixtures.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>
#include <portopt/logging.hpp>

int main(int argc, char* argv[]) {
    // Silence library logging during tests (use --log-level trace to enable)
    portopt::log::init(portopt::log::Level::Warn, true, true);

    return Catch::Session().run(argc, argv);
}
