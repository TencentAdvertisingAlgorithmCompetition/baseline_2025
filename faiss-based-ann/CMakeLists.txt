cmake_minimum_required(VERSION 3.10)
project(faiss_demo)

# 配置调试选项
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
if(build_type STREQUAL "DEBUG")
    message(STATUS "Building in DEBUG mode")
    
    # 通用调试标志
    add_compile_options(-g3 -O0 -fno-omit-frame-pointer)
    
    # 编译器特定选项
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-ggdb3 -fvar-tracking-assignments)
    endif()
endif()


# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置conda安装路径变量
# 获取CONDA_PREFIX环境变量
if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "CONDA_PREFIX environment variable not found! Please activate conda environment first.")
else()
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using conda prefix from environment: ${CONDA_PREFIX}")
endif()


# 设置包含路径变量
set(INCLUDE_DIRS
    ${CONDA_PREFIX}/include/
)

# 设置库文件路径变量
set(LINK_DIRS
    ${CONDA_PREFIX}/lib              # 库文件路径
)

# 添加可执行文件
add_executable(faiss_demo main.cc)
set_target_properties(faiss_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# 设置包含路径
target_include_directories(faiss_demo PRIVATE ${INCLUDE_DIRS})

# 设置链接库路径
target_link_directories(faiss_demo PRIVATE ${LINK_DIRS})

# 链接所需库
target_link_libraries(faiss_demo PRIVATE
    faiss    # Faiss库
    gflags   # gflags库
    stdc++fs
)
