cmake_minimum_required(VERSION 3.1)
project(cddp CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

set(MODEL_DIR ${PROJECT_SOURCE_DIR})
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/../../src)

add_library(
    ocp_model
    STATIC
    ${MODEL_DIR}/ocp_model.cpp
)

add_library(
    memory_manager
    STATIC
    ${SRC_DIR}/memory_manager.cpp
)
target_include_directories(
    memory_manager
    PRIVATE
    ${INCLUDE_DIR}
)

add_executable(
    a.out 
    ${MODEL_DIR}/main.cpp
)
target_include_directories(
    a.out
    PRIVATE
    ${MODEL_DIR}
    ${INCLUDE_DIR}
)
target_link_libraries(
    a.out
    PRIVATE
    ocp_model
    memory_manager
)
target_compile_options(
    a.out
    PRIVATE
    -O3
)