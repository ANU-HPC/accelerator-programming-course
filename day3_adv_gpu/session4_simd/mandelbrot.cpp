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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <liblsb.h>
#include <random>

const float EPSILON = 0.00001f;

inline void except(bool condition, const std::string &error_message = "") {
  if (!condition)
    throw std::runtime_error(error_message);
}

inline void zero_payload(float *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    x[i] = 0.0f;
  }
}

inline void randomise_payload(float *x, unsigned int size) {
  std::random_device seed;
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> dist(0, 100);

  for (int i = 0; i < size; i++) {
    x[i] = dist(gen);
  }
}

inline void copy_payload(float *in, float *out, unsigned int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}

bool same_payload(float *in, float *out, unsigned int size) {
  for (int i = 0; i < size; i++) {
    if (abs(out[i] - in[i]) > EPSILON) {
      return false;
    }
  }
  return true;
}

bool different_payload(float *in, float *out, unsigned int size) {
  return (!(same_payload(in, out, size)));
}

inline void print_payload(float *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl;
}

inline void print_payload_as_integer(int *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl;
}

inline void write_matrix_to_file(int *mat, unsigned int x, unsigned int y,
                                 std::string filename) {
  std::ofstream myfile;
  myfile.open(filename);
  for (int i = 0; i < y; i++) {       // rows
    for (int j = 0; j < x - 1; j++) { // cols
      myfile << mat[i * x + j] << ',';
    }
    myfile << mat[i * x + x] << '\n';
  }
  myfile.close();
}

inline void zero_payload_as_integer(int *x, unsigned int size) {
  for (int i = 0; i < size; i++) {
    x[i] = 0;
  }
}

void mandelbrot(int *map, int DIMX, int DIMY, float X_STEP, float Y_STEP) {
  int i, j;
  float x = -1.8f;
  for (i = 0; i < DIMX; i++) {
    float y = -0.2f;
    for (j = 0; j < DIMY / 2; j++) {
      int iter = 0;
      float sx = x;
      float sy = y;
      while (iter < 256) {
        if (sx * sx + sy * sy >= 4.0f) {
          break;
        }
        float old_sx = sx;
        sx = x + sx * sx - sy * sy;
        sy = y + 2 * old_sx * sy;
        iter++;
      }
      map[i * DIMY + j] = iter;
      y += Y_STEP;
    }
    x += X_STEP;
  }
}

#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

void mandelbrot_128(int *map, int DIMX, int DIMY, float X_STEP, float Y_STEP) {
  __m128 _INIT_Y_4 = {0, Y_STEP, 2 * Y_STEP, 3 * Y_STEP};
  __m128 _F_STEP_Y = {4 * Y_STEP, 4 * Y_STEP, 4 * Y_STEP, 4 * Y_STEP};
  __m128i _I_ONE = _mm_set1_epi32(1);
  __m128 _F_FOUR = {4.0f, 4.0f, 4.0f, 4.0f};
  __m128 _F_TWO = {2.0f, 2.0f, 2.0f, 2.0f};
  __m128 _F_STEP_X = {X_STEP, X_STEP, X_STEP, X_STEP};

  int i, j;
  __m128 x, y;
  for (i = 0, x = _mm_set1_ps(-1.8f); i < DIMX; i++, x = _mm_add_ps(x, _F_STEP_X)) {
    for (j = 0, y = _mm_add_ps(_mm_set1_ps(-0.2f), _INIT_Y_4); j < DIMY / 2; j += 4, y += _F_STEP_Y) {
      __m128 sx, sy;
      __m128i iter = _mm_setzero_si128();
      int scalar_iter = 0;
      sx = x;
      sy = y;
      while (scalar_iter < 256) {
        int mask = 0;
        __m128 old_sx = sx;
        __m128 vmask = _mm_cmpnlt_ps(sx * sx + sy * sy, _F_FOUR);
        // if all data points in our vector are hitting the "exit" condition,
        // the vectorized loop can exit
        if (_mm_test_all_ones(_mm_castps_si128(vmask)))
          break;
        // if none of the data points are out, we don’t need the extra code
        // which blends the results
        if (_mm_test_all_zeros(_mm_castps_si128(vmask), _mm_castps_si128(vmask))) {
          sx = x + sx * sx - sy * sy;
          sy = y + _F_TWO * old_sx * sy;
          iter += _I_ONE;
        } else {
          // Blended flavour of the code, this code blends values from previous
          // iteration with the values from current iteration. Only values
          // which did not hit the “exit” condition are being stored;
          // values which are already “out” are maintaining their value
          sx = _mm_blendv_ps(x + sx * sx - sy * sy, sx, vmask);
          sy = _mm_blendv_ps(y + _F_TWO * old_sx * sy, sy, vmask);
          iter = _mm_blendv_epi8(iter + _I_ONE, iter, _mm_castps_si128(vmask));
        }
        scalar_iter++;
      }
    }
  }
}

int main(int argc, char **argv) {
  except(argc == 1, "./mandelbrot");

  LSB_Init("mandelbrot", 0);
  LSB_Set_Rparam_string("kernel", "none_yet");

  LSB_Set_Rparam_string("region", "host_side_setup");
  LSB_Res();

  // set-up memory for payload/problem size
  size_t KiB = 7900;
  unsigned int c_bytes = (KiB * 1024);
  unsigned int c_elements =
      static_cast<unsigned int>(c_bytes / sizeof(unsigned int));
  // MxN matrix (but actually square matrix)
  int w = 32;
  int M = floor(sqrt(c_elements));
  M = floor(M / w) * w; // but rounded down so it's a multiple of 32 -- 32x32 divisible blocks

  unsigned int map_bytes = M * M * sizeof(int);

  std::cout << "M = " << M << " total KiB = " << map_bytes / 1024 << std::endl;

  LSB_Rec(0);

  std::cout << "Operating on a " << M << "x" << M << " map with a tile size "
            << w << "..." << std::endl;

  int dimx = M;
  int dimy = M;
  float xstep = 0.5f / dimx;
  float ystep = 0.4f / (dimy / 2);
  int *map = new int[M * M];

  int sample_size = 100;

  // mandelbrot case
  LSB_Set_Rparam_string("kernel", "mandelbrot");
  for (int i = 0; i < sample_size; i++) {
    LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
    zero_payload_as_integer(map, M * M);
    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "mandelbrot");
    LSB_Res();

    mandelbrot(map, dimx, dimy, xstep, ystep);
    LSB_Rec(i);
  }
  std::cout << "Mandelbrot:" << std::endl;
  write_matrix_to_file(map, M, M, "mandelbrot_set.csv");
  // print_payload_as_integer(map,M*M);
  // mandelbrot vectorized case
  LSB_Set_Rparam_string("kernel", "mandelbrot_vectorized");
  for (int i = 0; i < sample_size; i++) {
    LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
    zero_payload_as_integer(map, M * M);
    LSB_Rec(i);

    LSB_Set_Rparam_string("region", "mandelbrot_128");
    LSB_Res();

    mandelbrot_128(map, dimx, dimy, xstep, ystep);
    LSB_Rec(i);
  }
  std::cout << "\nMandelbrot vectorized:" << std::endl;
  write_matrix_to_file(map, M, M, "mandelbrot_set_vectorized.csv");
  // print_payload_as_integer(map,M*M);
  delete map;

  LSB_Finalize();
}
