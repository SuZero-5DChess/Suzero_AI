#pragma once
#include <chrono>
#include <stdexcept>
namespace profile {
using Clock=std::chrono::steady_clock;
inline double ms(Clock::duration d){return std::chrono::duration<double,std::milli>(d).count();}
inline Clock::time_point start;
inline double budget=5000,point_ms=0,problem_ms=0,remove_ms=0,isolate_ms=0,normalize_ms=0,destroy_ms=0,setup_ms=0;
inline size_t attempts=0,empty=0,problems=0;
inline void check(){if(ms(Clock::now()-start)>budget) throw std::runtime_error("budget");}
template<class F> decltype(auto) timed(double &counter,F &&f){
 auto t=Clock::now();
 if constexpr(std::is_void_v<decltype(f())>){f();counter+=ms(Clock::now()-t);}
 else {auto value=f();counter+=ms(Clock::now()-t);return value;}
}
}
