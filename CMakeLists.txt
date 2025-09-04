cmake_minimum_required(VERSION 3.12.0)
project(exam LANGUAGES CXX C)

include(cmake-common.cmake)

add_executable(main src/main.cpp src/FisherKolmogorov.cpp)

deal_ii_setup_target(main)

add_custom_target(remove_vtu_files
        COMMAND ${CMAKE_COMMAND} -E remove
        "${CMAKE_BINARY_DIR}/*.pvtu"
        "${CMAKE_BINARY_DIR}/*.vtu"
        "${CMAKE_BINARY_DIR}/output/*.pvtu"
        "${CMAKE_BINARY_DIR}/output/*.vtu"
        "*.pvtu"
        "*.vtu"
        "output/*.pvtu"
        "output/*.vtu"
        COMMENT "Removing all .pvtu and .vtu files from the build directory"
)

add_custom_target(create_output_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory output
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/output"
)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
        "${CMAKE_BINARY_DIR}/*.pvtu"
        "${CMAKE_BINARY_DIR}/*.vtu"
        "${CMAKE_BINARY_DIR}/output/*.pvtu"
        "${CMAKE_BINARY_DIR}/output/*.vtu"
        "*.pvtu"
        "*.vtu"
        "output/*.pvtu"
        "output/*.vtu"
)

add_dependencies(main remove_vtu_files create_output_dir)