#include "vk_app.hpp"

#include <cstdio>
#include <cstdlib>

int main() {
    try {
        VulkanApp app("Morphon -- SDF Ray Marching", 1280, 720);
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
