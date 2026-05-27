cmake_minimum_required(VERSION 3.25)
project(Morphon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ─── glslang (runtime shader compilation) ─────────────────────────────────
set(ALLOW_EXTERNAL_SPIRV_TOOLS OFF CACHE BOOL "" FORCE)
set(SKIP_GLSLANG_INSTALL    ON  CACHE BOOL "" FORCE)
set(ENABLE_OPT              OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL             OFF CACHE BOOL "" FORCE)
set(ENABLE_CTEST            OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING           OFF CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES  ON  CACHE BOOL "" FORCE)

FetchContent_Declare(
  glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
  GIT_TAG vulkan-sdk-1.3.296.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glslang)

# ─── Vulkan-Headers ──────────────────────────────────────────────────────
FetchContent_Declare(
  vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG v1.3.296
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vulkan_headers)

# ─── volk (Vulkan meta-loader) ───────────────────────────────────────────
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG vulkan-sdk-1.3.296
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

# ─── Shader sources dependency marker ────────────────────────────────────
file(GLOB_RECURSE SHADER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp
)
add_custom_target(shader_deps DEPENDS ${SHADER_SOURCES})

# ─── Main executable ─────────────────────────────────────────────────────
add_executable(Morphon
  src/main.cpp
  src/vk_app.cpp
  src/win32_window.cpp
)

target_compile_definitions(Morphon PRIVATE VK_USE_PLATFORM_WIN32_KHR)

target_link_libraries(Morphon PRIVATE
  Vulkan::Headers
  volk
  glslang::glslang
  glslang::SPIRV
  MachineIndependent
  GenericCodeGen
)

target_include_directories(Morphon PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${glslang_SOURCE_DIR}
  ${glslang_SOURCE_DIR}/SPIRV
)

# Re-link when shaders change
add_dependencies(Morphon shader_deps)

# Copy shader sources to build dir
add_custom_command(TARGET Morphon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Morphon>/shaders
)
foreach(SHADER ${SHADER_SOURCES})
  add_custom_command(TARGET Morphon POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER}
            $<TARGET_FILE_DIR:Morphon>/shaders/
  )
endforeach()
