# Morphon

**SDF Ray Marching Editor** — a real-time 3D sculpting tool built with Vulkan and Dear ImGui.

![Screenshot](images/Screenshot%202026-06-10%20180820.png)

## Features

- **Primitives:** Box, Sphere, Donut, Cylinder, Pyramid
- **Boolean Operations:** Union, Subtract, Intersect
- **Editor Tools:** Select, Marquee, Move, Rotate, Scale, Edit
- **Gizmo-based manipulation** with axis-constrained dragging
- **Face editing** for box primitives
- **Smooth blending** with configurable merge threshold
- **Orbit camera** with zoom
- **Multi-selection** and batch operations
- **Runtime shader compilation** via glslang

## Build

### Prerequisites

- CMake 3.25+
- C++20 compiler
- Vulkan SDK 1.3.296+
- Windows (Win32) or Linux (X11)

### Build

```sh
cmake -B build
cmake --build build
```

## Controls

| Action | Input |
|---|---|
| Orbit camera | Left-click + drag on empty area |
| Zoom | Scroll wheel |
| Select object | Click on object |
| Multi-select | Shift+click or marquee drag |
| Move object | Select → click gizmo axis + drag |
| Place primitive | Select Box/Sphere/etc. tool + click |
| Boolean operation | Select objects → choose op → Execute |

## Dependencies

- [Vulkan SDK](https://vulkan.lunarg.com/)
- [volk](https://github.com/zeux/volk) — Vulkan meta-loader
- [glslang](https://github.com/KhronosGroup/glslang) — runtime shader compilation
- [Dear ImGui](https://github.com/ocornut/imgui) — UI
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
