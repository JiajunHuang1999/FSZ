cmake_minimum_required(VERSION 3.18)

# Default GPU architectures. Users can override with
# -DCMAKE_CUDA_ARCHITECTURES=... ; sm_100 (Blackwell) is appended
# automatically when the CUDA toolkit is new enough to know it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
    set(_fsz_default_archs ON)
endif()

project(fsz LANGUAGES CXX CUDA VERSION 1.0.0)

if(_fsz_default_archs AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100)
endif()

if(CMAKE_CUDA_ARCHITECTURES MATCHES "^80(-real)?$")
    string(APPEND CMAKE_CUDA_FLAGS " -maxrregcount=96")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

option(FSZ_BUILD_SHARED   "Also build the shared library (libfsz.so)" ON)
option(FSZ_BUILD_TOOLS    "Build the fsz command-line tool"           ON)
option(FSZ_BUILD_EXAMPLES "Build FSZ examples"                        ON)
option(FSZ_BUILD_TESTS    "Build FSZ tests"                           ON)

set(FSZ_SOURCES src/fsz.cu)

# Static library (the default link target, exported as fsz::fsz).
add_library(fsz STATIC ${FSZ_SOURCES})
add_library(fsz::fsz ALIAS fsz)

# Shared library for dynamic linking and the Python bindings.
if(FSZ_BUILD_SHARED)
    add_library(fsz_shared SHARED ${FSZ_SOURCES})
    set_target_properties(fsz_shared PROPERTIES OUTPUT_NAME fsz)
endif()

foreach(_fsz_lib IN ITEMS fsz fsz_shared)
    if(NOT TARGET ${_fsz_lib})
        continue()
    endif()
    target_include_directories(${_fsz_lib}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${_fsz_lib} PUBLIC CUDA::cudart)
    set_target_properties(${_fsz_lib} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE  ON)
    target_compile_options(${_fsz_lib} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# Command-line tool: the binary is named `fsz`.
if(FSZ_BUILD_TOOLS)
    add_executable(fsz_cli tools/fsz.cu)
    set_target_properties(fsz_cli PROPERTIES OUTPUT_NAME fsz)
    target_include_directories(fsz_cli PRIVATE tools)
    target_link_libraries(fsz_cli PRIVATE fsz::fsz)
endif()

if(FSZ_BUILD_EXAMPLES)
    add_executable(fsz_example_roundtrip examples/example_roundtrip.cu)
    target_link_libraries(fsz_example_roundtrip PRIVATE fsz::fsz)

    add_executable(fsz_example_hostptr examples/example_hostptr.cu)
    target_link_libraries(fsz_example_hostptr PRIVATE fsz::fsz)
endif()

if(FSZ_BUILD_TESTS)
    enable_testing()
    add_executable(fsz_test_roundtrip tests/test_roundtrip.cu)
    target_link_libraries(fsz_test_roundtrip PRIVATE fsz::fsz)
    add_test(NAME roundtrip COMMAND fsz_test_roundtrip)

    add_executable(fsz_test_hostptr tests/test_hostptr.cu)
    target_include_directories(fsz_test_hostptr PRIVATE src)
    target_link_libraries(fsz_test_hostptr PRIVATE fsz::fsz)
    add_test(NAME hostptr COMMAND fsz_test_hostptr)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Fortran interface: built whenever a Fortran compiler is available.
option(FSZ_BUILD_FORTRAN "Build the Fortran interface" ON)
if(FSZ_BUILD_FORTRAN)
    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        add_library(fsz_fortran STATIC fortran/fsz.f90)
        target_link_libraries(fsz_fortran PUBLIC fsz::fsz stdc++)
        set_target_properties(fsz_fortran PROPERTIES
            Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules)
        target_include_directories(fsz_fortran INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>)
        if(FSZ_BUILD_EXAMPLES)
            add_executable(fsz_example_fortran fortran/example_fsz.f90)
            target_link_libraries(fsz_example_fortran PRIVATE fsz_fortran)
        endif()
        if(FSZ_BUILD_TESTS)
            add_executable(fsz_test_fortran fortran/test_fsz.f90)
            target_link_libraries(fsz_test_fortran PRIVATE fsz_fortran)
            add_test(NAME fortran COMMAND fsz_test_fortran)
        endif()
        install(TARGETS fsz_fortran
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/fsz.mod
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fsz)
    else()
        message(STATUS "FSZ: no Fortran compiler found; skipping the Fortran interface")
    endif()
endif()

install(TARGETS fsz EXPORT fszTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(FSZ_BUILD_SHARED)
    install(TARGETS fsz_shared EXPORT fszTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(FSZ_BUILD_TOOLS)
    install(TARGETS fsz_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT fszTargets
    FILE fszTargets.cmake
    NAMESPACE fsz::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fsz)

configure_package_config_file(cmake/fszConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fszConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fsz)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/fszConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/fszConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/fszConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fsz)
