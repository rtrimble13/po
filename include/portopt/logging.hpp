#pragma once
/**
 * @file logging.hpp
 * @brief Logging facade backed by spdlog.
 *
 * Provides a thin wrapper around spdlog so that library users can:
 *  - Set the global log level (trace → critical)
 *  - Add console and/or file sinks
 *  - Retrieve the named logger for library-internal use
 *
 * The library logger name is "portopt".  Client code may create their own
 * spdlog loggers; only calls to portopt::log::* affect the library logger.
 *
 * ### Typical setup
 * @code
 *   portopt::log::init();                   // console, info level
 *   portopt::log::setLevel(portopt::log::Level::Debug);
 *   portopt::log::addFileLog("portopt.log");
 *
 *   // Now run optimisations — all internal messages will be logged.
 * @endcode
 */

#include <spdlog/spdlog.h>
#include <memory>
#include <string>

namespace portopt {
namespace log {

/// Log severity levels (mirrors spdlog).
enum class Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off
};

/**
 * @brief Initialise the library logger.
 *
 * Safe to call multiple times; subsequent calls are no-ops unless
 * @p force_reinit is true.
 *
 * @param level         Initial log level
 * @param console       If true, attach a stderr/stdout sink
 * @param force_reinit  Destroy and re-create the logger
 */
void init(Level level      = Level::Info,
          bool  console    = true,
          bool  force_reinit = false);

/// Set the log level.
void setLevel(Level level);

/// Add a rotating file sink (max 5 MB, 3 files).
void addFileLog(const std::string& filename,
                Level              level = Level::Trace);

/// Return the raw spdlog logger (for advanced use).
std::shared_ptr<spdlog::logger> getLogger();

// ── Convenience macros (portopt-namespaced) ──────────────────────────────────

/// Log at TRACE level.
template<typename... Args>
void trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    getLogger()->trace(fmt, std::forward<Args>(args)...);
}

/// Log at DEBUG level.
template<typename... Args>
void debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    getLogger()->debug(fmt, std::forward<Args>(args)...);
}

/// Log at INFO level.
template<typename... Args>
void info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    getLogger()->info(fmt, std::forward<Args>(args)...);
}

/// Log at WARN level.
template<typename... Args>
void warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    getLogger()->warn(fmt, std::forward<Args>(args)...);
}

/// Log at ERROR level.
template<typename... Args>
void error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    getLogger()->error(fmt, std::forward<Args>(args)...);
}

} // namespace log
} // namespace portopt
