#ifndef _UTILS_TIMER_HPP
#define _UTILS_TIMER_HPP

#include <iostream>
#include <chrono>

struct Timer
{
    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::duration<float> duration;

    float* acc = nullptr;

    Timer()
    {
        this->start = std::chrono::high_resolution_clock::now();
    }

    Timer(float* acc)
        : acc(acc)
    {
        this->start = std::chrono::high_resolution_clock::now();
    }

    float ms()
    {

        this->end = std::chrono::high_resolution_clock::now();
        this->duration = this->end - this->start;

        return this->duration.count() * 1000.f;
    }

    float s()
    {

        this->end = std::chrono::high_resolution_clock::now();
        this->duration = this->end - this->start;

        return this->duration.count() * 1.f;
    }

    ~Timer()
    {
        float ms = this->ms();
        if(this->acc) { *acc += ms; }
        std::cout << "Time: " << ms << "ms" << std::endl;
    }
};

#endif