add_library(portopt_lib STATIC
    logging.cpp
    qp_solver.cpp
    mvo.cpp
    black_litterman.cpp
    estimation.cpp
    io/reader.cpp
    io/writer.cpp
)

target_include_directories(portopt_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(portopt_lib
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    PRIVATE
        tomlplusplus::tomlplusplus
)

# Strict warnings
if(MSVC)
    target_compile_options(portopt_lib PRIVATE /W4 /WX)
else()
    target_compile_options(portopt_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
