#pragma once

#include "win32_window.hpp"
#include <volk.h>

#include <array>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

enum class PrimType {
    Box = 0,
    Sphere = 1,
    Donut = 2,
    Cylinder = 3,
    Pyramid = 4
};

enum class ToolType {
    Select,   // click to select, green highlight
    Marquee,  // drag marquee to select
    Move,     // click-drag to move
    Rotate,   // rotate tool (TODO)
    Scale,    // scale tool (TODO)
    Edit,     // edit selected object properties
    Box, Sphere, Donut, Cylinder, Pyramid  // placement tools
};

enum class OpType {
    Union = 0,
    Subtract = 1,
    Intersect = 2
};

struct PlacedObject {
    float type;       // 0 = additive, 1 = subtractive
    float primType;   // 0=Box, 1=Sphere, 2=Donut, 3=Cylinder, 4=Pyramid
    float param1;     // shape-specific param (box-half-extent, sphere-r, donut-R, cyl-h, pyr-h)
    float param2;     // shape-specific param (box-half-extent-z, donut-r, cyl-r, pyr-r)
    float px, py, pz, hidden; // position + hidden flag
    float rx, ry, rz;         // rotation (Euler angles in radians)
    float _rotPad;            // padding to match GLSL vec4 alignment
    float sx, sy, sz;         // scale factors
    float _sclPad;            // padding to match GLSL vec4 alignment
};
static_assert(sizeof(PlacedObject) == 64, "PlacedObject must be 64 bytes");

static constexpr int MAX_OBJECTS = 256;

class VulkanApp {
public:
    VulkanApp(const std::string& title, int width, int height);
    ~VulkanApp();
    void run();

private:
    // Window
    Win32Window* window_ = nullptr;
    int width_ = 0, height_ = 0;
    bool resized_ = false;
    std::chrono::steady_clock::time_point startTime_;

    // Editor state
    ToolType activeTool_ = ToolType::Select;
    bool addAsVoid_ = false;    // true to place subtractive (void)
    bool snapEnabled_ = false;  // snap to half-grid when placing
    float mergeThreshold_ = 0.05f; // smooth-union blend radius
    std::vector<PlacedObject> placedObjects_;
    bool lastLeftDown_ = false;
    int  clickStartX_ = 0, clickStartY_ = 0;

    // Selection
    int  selectedIndex_ = -1;
    std::vector<int> selectedIndices_;

    // Marquee select state
    bool marqueeActive_ = false;
    int  marqueeStartX_ = 0, marqueeStartY_ = 0;
    int  marqueeEndX_ = 0, marqueeEndY_ = 0;

    // Boolean operation
    OpType opType_ = OpType::Union;

    // Hidden objects (bitset)
    std::bitset<MAX_OBJECTS> hiddenFlags_;

    // Ghost preview (CPU ray-march result, synced with GPU)
    float ghostPosX_ = 0.0f, ghostPosY_ = 0.0f, ghostPosZ_ = 0.0f;
    bool ghostValid_ = false;
    int  ghostMouseX_ = -1, ghostMouseY_ = -1;
    ToolType ghostTool_ = ToolType::Select;
    bool ghostVoid_ = false;

    // Selection & gizmo
    int  gizmoAxis_ = 0;  // 0=none, 1=X, 2=Y, 3=Z
    bool isDragging_ = false;
    int  dragStartSX_ = 0, dragStartSY_ = 0;

    // Drag start state for position
    float dragObjX_ = 0, dragObjY_ = 0, dragObjZ_ = 0;
    std::vector<float> dragOrigX_, dragOrigY_, dragOrigZ_;

    // Drag start state for rotation
    float dragObjRx_ = 0, dragObjRy_ = 0, dragObjRz_ = 0;
    std::vector<float> dragOrigRx_, dragOrigRy_, dragOrigRz_;

    // Drag start state for scale
    float dragObjSx_ = 0, dragObjSy_ = 0, dragObjSz_ = 0;
    std::vector<float> dragOrigSx_, dragOrigSy_, dragOrigSz_;

    // Face editing drag state
    int faceDragAxis_ = 0;    // 0=none, 1=X, 2=Y, 3=Z
    float faceDragSign_ = 0;  // +1 or -1 for face direction
    float faceDragOrigSx_ = 1.0f, faceDragOrigSy_ = 1.0f, faceDragOrigSz_ = 1.0f;
    int faceDragStartX_ = 0, faceDragStartY_ = 0;

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
    VkRenderPass imguiRenderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Gizmo mesh pipeline
    VkPipelineLayout gizmoPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gizmoPipeline_ = VK_NULL_HANDLE;

    // Gizmo vertex/index buffers
    struct GizmoMesh {
        VkBuffer vertexBuf = VK_NULL_HANDLE;
        VkDeviceMemory vertexMem = VK_NULL_HANDLE;
        VkBuffer indexBuf = VK_NULL_HANDLE;
        VkDeviceMemory indexMem = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
    };
    GizmoMesh gizmoMove_;
    GizmoMesh gizmoRotate_;
    GizmoMesh gizmoScale_;

    // Descriptors
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorPool imguiDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    // UBO (binding 0)
    VkBuffer ubo_ = VK_NULL_HANDLE;
    VkDeviceMemory uboMem_ = VK_NULL_HANDLE;

    // SSBO for placed objects (binding 1)
    VkBuffer objectBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory objectMem_ = VK_NULL_HANDLE;

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
        float mouseNdcX;
        float mouseNdcY;
        int32_t editorMode;
        alignas(16) std::array<float, 4> cameraPos;
        alignas(16) std::array<float, 4> cameraTarget;
        alignas(16) std::array<float, 4> ghostPos; // xyz = position, w = valid flag
        alignas(16) std::array<float, 4> ghostPrimInfo; // x=primType, y=param1, z=param2, w=unused
        alignas(16) std::array<float, 4> selectedPos; // xyz = position, w = valid flag
        alignas(16) std::array<float, 4> selectedPrimInfo; // x=primType, y=param1, z=param2, w=showGizmo
        alignas(16) std::array<float, 4> camRight;    // camera right direction
        alignas(16) std::array<float, 4> camUp;       // camera up direction
        alignas(16) std::array<uint32_t, 8> hiddenFlags; // bitset for 256 objects

    // Multi-selection highlight
    int32_t selectedCount;
    float _padMsel[3];
    alignas(16) std::array<float, 4> selPos[32];
    alignas(16) std::array<float, 4> selInfo[32];

    // Projection * view matrix for mesh gizmo rendering
    alignas(16) std::array<float, 16> viewProj;
};

    // Init helpers
    void initWindow();
    void initVulkan();
    void cleanup();

    // Gizmo mesh helpers
    void createGizmoPipeline();
    void createGizmoBuffers();
    void destroyGizmoBuffers();
    void drawGizmo(VkCommandBuffer cmd, VkBuffer vtxBuf, VkBuffer idxBuf,
                   uint32_t idxCount, const std::array<float, 3>& center,
                   ToolType tool);

    void createInstance();
    void setupDebug();
    void createSurface();
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createRenderPass();
    void createDescriptors();
    void createUniformBuffer();
    void createObjectBuffer();
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
    void handleInput();
    void placeObject();
    static float cpuSDF(const std::array<float, 3>& p, const PlacedObject& obj);
    bool cpuRayMarch(int screenX, int screenY, float& hx, float& hy, float& hz);
    int  pickObject(int screenX, int screenY);
    int  gizmoHitTest(int screenX, int screenY);
    void startGizmoDrag(int screenX, int screenY);
    void doGizmoDrag(int screenX, int screenY);
    void doFaceDrag(int screenX, int screenY);
    void executeBooleanOp();
    void updateTitle();
    void updateObjectBuffer();
    void drawFrame();

    void initImgui();
    void renderImgui();

    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
};
