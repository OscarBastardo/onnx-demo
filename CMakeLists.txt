cmake_minimum_required(VERSION 3.10)
project(onnx_demo_cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Path to ONNX Runtime installation ---
# Use a relative path to the folder inside the project directory
set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

# Find the include directory
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOTDIR}/include)

# Find the library file
find_library(ONNXRUNTIME_LIB onnxruntime
             HINTS ${ONNXRUNTIME_ROOTDIR}/lib)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Ensure the 'onnxruntime' directory is next to this CMakeLists.txt.")
endif()

# Add the executable
add_executable(onnx_demo main.cpp)

# Link the library and include directories to the target
target_include_directories(onnx_demo PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(onnx_demo PRIVATE ${ONNXRUNTIME_LIB})

# Set the RPATH to tell the compiled executable where to find the .dylib file at runtime.
if(APPLE)
    # Set the rpath to be the executable's own directory
    set(CMAKE_INSTALL_RPATH "@executable_path")

    # Add a post-build command to copy the .dylib next to the executable
    add_custom_command(TARGET onnx_demo POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${ONNXRUNTIME_LIB}
                       $<TARGET_FILE_DIR:onnx_demo>)
endif()