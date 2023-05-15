# CMakeList.txt : CMake project for CppGPT, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


project ("CppGPT")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Libtorch/libtorch-win-shared-with-deps-debug-2.0.1+cu117/libtorch")
else()
  set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Libtorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch")
endif()


message("${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source to this project's executable.
add_executable (CppGPT "CppGPT.cpp" "CppGPT.h")
target_link_libraries(CppGPT "${TORCH_LIBRARIES}")
target_include_directories(CppGPT PUBLIC "${TORCH_INCLUDE_DIRS}")
message("${TORCH_INCLUDE_DIRS}")
set_property(TARGET CppGPT PROPERTY CXX_STANDARD 14)
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CppGPT PROPERTY CXX_STANDARD 20)
endif()
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CppGPT
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CppGPT>)
endif (MSVC)
# TODO: Add tests and install targets if needed.
