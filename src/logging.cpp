#include <portopt/logging.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <mutex>

namespace portopt {
namespace log {

static std::mutex        g_mutex;
static bool              g_initialised = false;

static spdlog::level::level_enum toSpdlog(Level lvl) {
    switch (lvl) {
        case Level::Trace:    return spdlog::level::trace;
        case Level::Debug:    return spdlog::level::debug;
        case Level::Info:     return spdlog::level::info;
        case Level::Warn:     return spdlog::level::warn;
        case Level::Error:    return spdlog::level::err;
        case Level::Critical: return spdlog::level::critical;
        default:              return spdlog::level::off;
    }
}

void init(Level level, bool console, bool force_reinit) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialised && !force_reinit)
        return;

    std::vector<spdlog::sink_ptr> sinks;

    if (console) {
        auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        sink->set_level(toSpdlog(level));
        sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [portopt] [%^%l%$] %v");
        sinks.push_back(std::move(sink));
    }

    auto logger = std::make_shared<spdlog::logger>("portopt",
                                                    sinks.begin(), sinks.end());
    logger->set_level(toSpdlog(level));
    logger->flush_on(spdlog::level::warn);

    spdlog::drop("portopt");
    spdlog::register_logger(logger);
    g_initialised = true;
}

void setLevel(Level level) {
    getLogger()->set_level(toSpdlog(level));
}

void addFileLog(const std::string& filename, Level level) {
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        filename, 5 * 1024 * 1024 /* 5 MB */, 3);
    file_sink->set_level(toSpdlog(level));
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [portopt] [%l] %v");
    getLogger()->sinks().push_back(file_sink);
}

std::shared_ptr<spdlog::logger> getLogger() {
    if (!g_initialised) {
        init();
    }
    auto logger = spdlog::get("portopt");
    if (!logger) {
        init(Level::Info, true, true);
        logger = spdlog::get("portopt");
    }
    return logger;
}

} // namespace log
} // namespace portopt
