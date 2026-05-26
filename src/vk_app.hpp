#pragma once

#include "x11_window.hpp"
#include <volk.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

class VulkanApp {
public:
    VulkanApp(const std::string& title, int width, int height);
    ~VulkanApp();
    void run();

private:
    // Window
    X11Window* window_ = nullptr;
    int width_ = 0, height_ = 0;
    bool resized_ = false;
    std::chrono::steady_clock::time_point startTime_;

    // Vulkan core
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT dbg_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_ = VK_NULL_HANDLE;
    uint32_t graphicsQ_ = 0, presentQ_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_ = {};

    // Pipeline
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Descriptors
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    // UBO
    VkBuffer ubo_ = VK_NULL_HANDLE;
    VkDeviceMemory uboMem_ = VK_NULL_HANDLE;

    // Per-frame data
    struct Frame {
        VkImage image = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };
    std::vector<Frame> frames_;
    uint32_t currentFrame_ = 0;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    // Camera orbit
    float camTheta_ = 0.0f;
    float camPhi_ = 0.35f;
    float camDist_ = 6.0f;
    std::array<float, 3> camTarget_{0.0f, 0.0f, 0.0f};

    // Uniform buffer data
    struct UBO {
        float time;
        float pad[3];
        alignas(16) std::array<float, 4> cameraPos;
        alignas(16) std::array<float, 4> cameraTarget;
    };

    // Init helpers
    void initWindow();
    void initVulkan();
    void cleanup();

    void createInstance();
    void setupDebug();
    void createSurface();
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createRenderPass();
    void createDescriptors();
    void createUniformBuffer();
    VkShaderModule compileShader(const std::string& path, VkShaderStageFlagBits stage);
    void createPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();

    // Runtime
    void rebuildSwapchain();
    void recordCommandBuffer(Frame& frame, uint32_t imageIndex);
    void updateUniforms();
    void drawFrame();

    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
};
