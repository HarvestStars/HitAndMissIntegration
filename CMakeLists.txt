# cmake -B [build directory] -S [source directory]
cmake_minimum_required(VERSION 3.10)

# set the project name
project(ortho_sampling_generate C)

# source files
set(SOURCE_FILES
    ortho-pack/ortho_sampling_generate.c
    ortho-pack/mt19937ar.c
    ortho-pack/rand_support.c
)

# add dynamic library(SHARED)
add_library(ortho_sampling_generate SHARED ${SOURCE_FILES})

set_target_properties(ortho_sampling_generate PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ortho-pack/lib
)

# corss platform settings
if(WIN32)
    # Windows
    set_target_properties(ortho_sampling_generate PROPERTIES SUFFIX ".dll")
else()
    # Linux
    set_target_properties(ortho_sampling_generate PROPERTIES SUFFIX ".so")
endif()
