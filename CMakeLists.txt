cmake_minimum_required (VERSION 3.11)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


project ("CppGPT")
if (MSVC)
  # if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  # unsure how to properly do this, this does at least not work
  set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/Libtorch/libtorch-win-shared-with-deps-debug-2.0.1+cu117/libtorch")
  # else()
  # set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/Libtorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch")
  # endif()
else()
  set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Libtorch/libtorch")
endif(MSVC)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        57a9fd0841f00e92b478a07fef62636d7be612a8 # release-1.10.0
)

FetchContent_MakeAvailable(spdlog)

if (MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W4")
endif(MSVC)

add_executable (CppGPT "CppGPT.cpp" "CppGPT.h")
set(SPDLOG_FMT_EXTERNAL ON)
target_link_libraries(CppGPT 
                            PRIVATE "${TORCH_LIBRARIES}"
                            PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
target_include_directories(CppGPT 
                                  PUBLIC "${TORCH_INCLUDE_DIRS}"
                                  PRIVATE "${SPDLOG_INCLUDE_DIRS}")

set_property(TARGET CppGPT PROPERTY CXX_STANDARD 20)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CppGPT
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CppGPT>)
endif (MSVC)
