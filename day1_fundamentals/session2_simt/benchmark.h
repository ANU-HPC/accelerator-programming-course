/*
 * Copyright 2019 Australian National University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either or express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <sys/time.h>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>

/** Get CPU time in microseconds */
int64_t timeInMicros() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

const int ITERS = 100;

/**
 * Measure the average time taken to execute a function,
 * by measuring the time taken to execute the function
 * a number of times inside a loop.
 */
void measureFunctionTime(std::function<void()> func, std::string funcName) {
  int64_t startTime = timeInMicros();

  for (int i = 0; i < ITERS; i++) {
    func();
  }

  int64_t endTime = timeInMicros();

  printf("%s took %.2f ms\n", funcName.c_str(),
         (endTime - startTime) / ITERS / 1e3);
}

/**
 * Check that the result of some calculation was correct within a tiny
 * relative error.
 */
bool checkWithinTolerance(double expected, double actual) {
  if (std::abs(actual - expected) / expected > 1e-12) {
    printf("Incorrect result: expected %.2e but got %.2e\n", expected, actual);
  }
}

#endif  // _BENCHMARK_H
