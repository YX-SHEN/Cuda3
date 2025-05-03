#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

class Timer {
public:
    Timer() : start_time({0, 0}) {}

    void start() {
        gettimeofday(&start_time, NULL);
    }

    double elapsed() const {
        timeval end_time;
        gettimeofday(&end_time, NULL);
        return (end_time.tv_sec - start_time.tv_sec) +
               (end_time.tv_usec - start_time.tv_usec) * 1e-6;
    }

private:
    timeval start_time;
};

#endif
