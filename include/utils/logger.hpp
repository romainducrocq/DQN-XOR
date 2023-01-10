#ifndef _UTILS_LOGGER_HPP
#define _UTILS_LOGGER_HPP

#include <iostream>

namespace Logger
{
    inline void fatal(const std::string& msg){
        std::cerr << "[FATAL] " << msg << "\n";
    }

    inline void error(const std::string& msg){
        std::cerr << "[ERROR] " << msg << "\n";
    }

    inline void warn(const std::string& msg){
        std::cerr << "[WARN]  " << msg << "\n";
    }

    inline void info(const std::string& msg){
        std::cerr << "[INFO]  " << msg << "\n";
    }

    inline void debug(const std::string& msg){
        std::cerr << "[DEBUG] " << msg << "\n";
    }

    inline void trace(const std::string& msg){
        std::cerr << "[TRACE] " << msg << "\n";
    }

}

#endif
