cmake_minimum_required(VERSION 3.10)

project(NTT_Demo VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 开启 AVX2 和 O3 优化 ---
if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    add_compile_options(-mavx2 -mfma -O3 -march=native)
endif()

include_directories(include)

add_library(ntt_core src/ntt.cpp)
add_executable(ntt_test tests/test_main.cpp)
target_link_libraries(ntt_test PRIVATE ntt_core)