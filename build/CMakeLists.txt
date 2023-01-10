cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(exec)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -std=c++17 ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/../src/")
set(HEADER_DIR "${PROJECT_SOURCE_DIR}/../include/")
set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../test/src/")
set(TEST_HEADER_DIR "${PROJECT_SOURCE_DIR}/../test/include/")
set(LIB_DIR "${PROJECT_SOURCE_DIR}/../lib/")
set(EXEC_DIR "${PROJECT_SOURCE_DIR}/../bin/apps/")
set(LINK_LIB "-lgtest -pthread")

file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIR}/*.cpp" "${TEST_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE HEADER_FILES "${HEADER_DIR}/*.hpp" "${TEST_HEADER_DIR}/*.hpp" "${LIB_DIR}/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${TEST_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${HEADER_DIR} ${TEST_HEADER_DIR} ${LIB_DIR})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${LINK_LIB})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXEC_DIR})
