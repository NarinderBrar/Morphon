#include "vk_app.hpp"

#define VK_NO_PROTOTYPES
#include <volk.h>

#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

// ── Vulkan debug callback ────────────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::fprintf(stderr, "[VULKAN] %s\n", data->pMessage);
    return VK_FALSE;
}

static std::string readTextFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to read: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── glslang built-in resources ───────────────────────────────────────────

static const TBuiltInResource defaultResources = {
    .maxLights = 32,
    .maxClipPlanes = 6,
    .maxTextureUnits = 32,
    .maxTextureCoords = 32,
    .maxVertexAttribs = 64,
    .maxVertexUniformComponents = 4096,
    .maxVaryingFloats = 64,
    .maxVertexTextureImageUnits = 32,
    .maxCombinedTextureImageUnits = 80,
    .maxTextureImageUnits = 32,
    .maxFragmentUniformComponents = 4096,
    .maxDrawBuffers = 32,
    .maxVertexUniformVectors = 128,
    .maxVaryingVectors = 8,
    .maxFragmentUniformVectors = 16,
    .maxVertexOutputVectors = 16,
    .maxFragmentInputVectors = 15,
    .minProgramTexelOffset = -8,
    .maxProgramTexelOffset = 7,
    .maxClipDistances = 8,
    .maxComputeWorkGroupCountX = 65535,
    .maxComputeWorkGroupCountY = 65535,
    .maxComputeWorkGroupCountZ = 65535,
    .maxComputeWorkGroupSizeX = 1024,
    .maxComputeWorkGroupSizeY = 1024,
    .maxComputeWorkGroupSizeZ = 64,
    .maxComputeUniformComponents = 1024,
    .maxComputeTextureImageUnits = 16,
    .maxComputeImageUniforms = 8,
    .maxComputeAtomicCounters = 8,
    .maxComputeAtomicCounterBuffers = 1,
    .maxVaryingComponents = 60,
    .maxVertexOutputComponents = 64,
    .maxGeometryInputComponents = 64,
    .maxGeometryOutputComponents = 128,
    .maxFragmentInputComponents = 128,
    .maxImageUnits = 8,
    .maxCombinedImageUnitsAndFragmentOutputs = 8,
    .maxCombinedShaderOutputResources = 8,
    .maxImageSamples = 0,
    .maxVertexImageUniforms = 0,
    .maxTessControlImageUniforms = 0,
    .maxTessEvaluationImageUniforms = 0,
    .maxGeometryImageUniforms = 0,
    .maxFragmentImageUniforms = 8,
    .maxCombinedImageUniforms = 8,
    .maxGeometryTextureImageUnits = 16,
    .maxGeometryOutputVertices = 256,
    .maxGeometryTotalOutputComponents = 1024,
    .maxGeometryUniformComponents = 1024,
    .maxGeometryVaryingComponents = 64,
    .maxTessControlInputComponents = 128,
    .maxTessControlOutputComponents = 128,
    .maxTessControlTextureImageUnits = 16,
    .maxTessControlUniformComponents = 1024,
    .maxTessControlTotalOutputComponents = 4096,
    .maxTessEvaluationInputComponents = 128,
    .maxTessEvaluationOutputComponents = 128,
    .maxTessEvaluationTextureImageUnits = 16,
    .maxTessEvaluationUniformComponents = 1024,
    .maxTessPatchComponents = 120,
    .maxPatchVertices = 32,
    .maxTessGenLevel = 64,
    .maxViewports = 16,
    .maxVertexAtomicCounters = 0,
    .maxTessControlAtomicCounters = 0,
    .maxTessEvaluationAtomicCounters = 0,
    .maxGeometryAtomicCounters = 0,
    .maxFragmentAtomicCounters = 8,
    .maxCombinedAtomicCounters = 8,
    .maxAtomicCounterBindings = 1,
    .maxVertexAtomicCounterBuffers = 0,
    .maxTessControlAtomicCounterBuffers = 0,
    .maxTessEvaluationAtomicCounterBuffers = 0,
    .maxGeometryAtomicCounterBuffers = 0,
    .maxFragmentAtomicCounterBuffers = 1,
    .maxCombinedAtomicCounterBuffers = 1,
    .maxAtomicCounterBufferSize = 16384,
    .maxTransformFeedbackBuffers = 4,
    .maxTransformFeedbackInterleavedComponents = 64,
    .maxCullDistances = 8,
    .maxCombinedClipAndCullDistances = 8,
    .maxSamples = 4,
    .maxMeshOutputVerticesNV = 256,
    .maxMeshOutputPrimitivesNV = 512,
    .maxMeshWorkGroupSizeX_NV = 32,
    .maxMeshWorkGroupSizeY_NV = 1,
    .maxMeshWorkGroupSizeZ_NV = 1,
    .maxTaskWorkGroupSizeX_NV = 32,
    .maxTaskWorkGroupSizeY_NV = 1,
    .maxTaskWorkGroupSizeZ_NV = 1,
    .maxMeshViewCountNV = 4,
    .maxMeshOutputVerticesEXT = 256,
    .maxMeshOutputPrimitivesEXT = 512,
    .maxMeshWorkGroupSizeX_EXT = 32,
    .maxMeshWorkGroupSizeY_EXT = 1,
    .maxMeshWorkGroupSizeZ_EXT = 1,
    .maxTaskWorkGroupSizeX_EXT = 32,
    .maxTaskWorkGroupSizeY_EXT = 1,
    .maxTaskWorkGroupSizeZ_EXT = 1,
    .maxMeshViewCountEXT = 4,
    .maxDualSourceDrawBuffersEXT = 1,
};

static EShLanguage vkStageToShaderc(VkShaderStageFlagBits stage) {
    switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:   return EShLangVertex;
    case VK_SHADER_STAGE_FRAGMENT_BIT: return EShLangFragment;
    case VK_SHADER_STAGE_COMPUTE_BIT:  return EShLangCompute;
    default: throw std::runtime_error("Unsupported shader stage");
    }
}

// ── ctor / dtor ──────────────────────────────────────────────────────────

VulkanApp::VulkanApp(const std::string& title, int width, int height)
    : width_(width), height_(height)
{
    initWindow();
    initVulkan();
}

VulkanApp::~VulkanApp() { cleanup(); }

// ── run ──────────────────────────────────────────────────────────────────

void VulkanApp::run() {
    startTime_ = std::chrono::steady_clock::now();
    while (!window_->shouldClose()) {
        window_->pollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(device_);
}

// ── init window ──────────────────────────────────────────────────────────

void VulkanApp::initWindow() {
    window_ = new X11Window("Morphon -- SDF Ray Marching", width_, height_);
}

// ── init vulkan ──────────────────────────────────────────────────────────

void VulkanApp::initVulkan() {
    glslang::InitializeProcess();

    volkInitialize();

    createInstance();
    setupDebug();
    createSurface();
    pickPhysicalDevice();
    createDevice();
    createSwapchain();
    createRenderPass();
    createDescriptors();
    createUniformBuffer();
    createPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

// ── instance ─────────────────────────────────────────────────────────────

static bool checkValidationLayerSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (auto& l : layers)
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            return true;
    return false;
}

void VulkanApp::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Morphon";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName        = "Morphon";
    appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    auto exts = X11Window::getVulkanExtensions();

    // Check which extensions are available
    uint32_t extCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availExts.data());
    std::set<std::string> availExtSet;
    for (auto& e : availExts) availExtSet.insert(e.extensionName);

    bool debugAvailable = availExtSet.count(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugAvailable) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::fprintf(stdout, "[INFO] VK_EXT_debug_utils available\n");
    } else {
        std::fprintf(stderr, "[WARN] VK_EXT_debug_utils not available\n");
    }

    bool valAvailable = checkValidationLayerSupport();
    const char* layers[1]{};
    uint32_t layerCount = 0;
    if (valAvailable) {
        layers[0] = "VK_LAYER_KHRONOS_validation";
        layerCount = 1;
        std::fprintf(stdout, "[INFO] Validation layers enabled\n");
    } else {
        std::fprintf(stderr, "[WARN] VK_LAYER_KHRONOS_validation not available\n");
    }

    VkInstanceCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo        = &appInfo;
    info.enabledExtensionCount   = static_cast<uint32_t>(exts.size());
    info.ppEnabledExtensionNames = exts.data();
    info.enabledLayerCount       = layerCount;
    info.ppEnabledLayerNames     = layerCount ? layers : nullptr;

    VkResult res = vkCreateInstance(&info, nullptr, &instance_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create instance");
    volkLoadInstance(instance_);
}

void VulkanApp::setupDebug() {
    VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
    dbgInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbgInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbgInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbgInfo.pfnUserCallback = debugCallback;
    VkResult res = vkCreateDebugUtilsMessengerEXT(instance_, &dbgInfo, nullptr, &dbg_);
    if (res != VK_SUCCESS)
        std::fprintf(stderr, "Warning: debug messenger not available\n");
}

void VulkanApp::createSurface() {
    surface_ = window_->createVulkanSurface(instance_);
}

// ── physical device ──────────────────────────────────────────────────────

void VulkanApp::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan-capable GPU found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    for (auto d : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            continue;

        uint32_t qc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, qp.data());

        int graphics = -1, present = -1;
        for (uint32_t i = 0; i < qc; ++i) {
            if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = static_cast<int>(i);
            VkBool32 supports = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &supports);
            if (supports) present = static_cast<int>(i);
        }
        if (graphics >= 0 && present >= 0) {
            physDev_   = d;
            graphicsQ_ = static_cast<uint32_t>(graphics);
            presentQ_  = static_cast<uint32_t>(present);
            return;
        }
    }
    throw std::runtime_error("No suitable GPU found");
}

// ── logical device ──────────────────────────────────────────────────────

void VulkanApp::createDevice() {
    std::set<uint32_t> unique = {graphicsQ_, presentQ_};
    std::vector<VkDeviceQueueCreateInfo> qcis;
    float priority = 1.0f;
    for (auto q : unique) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = q;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &priority;
        qcis.push_back(qci);
    }

    const char* exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount    = static_cast<uint32_t>(qcis.size());
    info.pQueueCreateInfos       = qcis.data();
    info.enabledExtensionCount   = 1;
    info.ppEnabledExtensionNames = exts;
    info.pEnabledFeatures        = &features;

    VkResult res = vkCreateDevice(physDev_, &info, nullptr, &device_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create device");
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, graphicsQ_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentQ_, 0, &presentQueue_);
}

// ── swapchain ────────────────────────────────────────────────────────────

void VulkanApp::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev_, surface_, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev_, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev_, surface_, &fmtCount, fmts.data());

    VkSurfaceFormatKHR fmt = fmts[0];
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) {
            fmt = f;
            break;
        }
    }
    swapchainFormat_ = fmt.format;

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    {
        uint32_t pmc = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physDev_, surface_, &pmc, nullptr);
        std::vector<VkPresentModeKHR> pms(pmc);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physDev_, surface_, &pmc, pms.data());
        for (auto pm : pms) {
            if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
                presentMode = pm;
                break;
            }
        }
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == std::numeric_limits<uint32_t>::max()) {
        extent.width  = std::clamp(static_cast<uint32_t>(width_),
            caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<uint32_t>(height_),
            caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    swapchainExtent_ = extent;
    width_  = static_cast<int>(extent.width);
    height_ = static_cast<int>(extent.height);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = surface_;
    sci.minImageCount    = imageCount;
    sci.imageFormat      = fmt.format;
    sci.imageColorSpace  = fmt.colorSpace;
    sci.imageExtent      = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (graphicsQ_ != presentQ_) {
        sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        uint32_t qs[] = {graphicsQ_, presentQ_};
        sci.pQueueFamilyIndices   = qs;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    sci.preTransform   = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode    = presentMode;
    sci.clipped        = VK_TRUE;

    VkResult res = vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create swapchain");

    uint32_t ic = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &ic, nullptr);
    std::vector<VkImage> images(ic);
    vkGetSwapchainImagesKHR(device_, swapchain_, &ic, images.data());

    frames_.resize(ic);
    for (uint32_t i = 0; i < ic; ++i) {
        frames_[i].image = images[i];
        VkImageViewCreateInfo ivci{};
        ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image    = images[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format   = fmt.format;
        ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.levelCount     = 1;
        ivci.subresourceRange.layerCount     = 1;
        res = vkCreateImageView(device_, &ivci, nullptr, &frames_[i].imageView);
        if (res != VK_SUCCESS) throw std::runtime_error("Failed to create image view");
    }
}

// ── render pass ──────────────────────────────────────────────────────────

void VulkanApp::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = swapchainFormat_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &color;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;

    VkResult res = vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
}

// ── descriptors ──────────────────────────────────────────────────────────

void VulkanApp::createDescriptors() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1;
    dslci.pBindings    = &binding;
    VkResult res = vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &descSetLayout_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create descriptor set layout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &poolSize;

    res = vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create descriptor pool");

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &descSetLayout_;

    res = vkAllocateDescriptorSets(device_, &dsai, &descSet_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to allocate descriptor set");
}

// ── uniform buffer ───────────────────────────────────────────────────────

void VulkanApp::createUniformBuffer() {
    VkDeviceSize sz = sizeof(UBO);

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = sz;
    bci.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult res = vkCreateBuffer(device_, &bci, nullptr, &ubo_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create UBO");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, ubo_, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    res = vkAllocateMemory(device_, &mai, nullptr, &uboMem_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to allocate UBO memory");

    vkBindBufferMemory(device_, ubo_, uboMem_, 0);

    VkDescriptorBufferInfo dbi{};
    dbi.buffer = ubo_;
    dbi.range  = sz;

    VkWriteDescriptorSet wds{};
    wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet          = descSet_;
    wds.dstBinding      = 0;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wds.pBufferInfo     = &dbi;
    vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);
}

// ── shader compilation via glslang ──────────────────────────────────────

VkShaderModule VulkanApp::compileShader(const std::string& path, VkShaderStageFlagBits stage) {
    std::string source = readTextFile(path);
    const char* srcCStr = source.c_str();

    EShLanguage lang = vkStageToShaderc(stage);

    glslang::TShader shader(lang);
    shader.setStrings(&srcCStr, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang,
                       glslang::EShClientVulkan, 460);
    shader.setEnvClient(glslang::EShClientVulkan,
                        glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv,
                        glslang::EShTargetSpv_1_6);

    if (!shader.parse(&defaultResources, 460, false, EShMsgDefault)) {
        std::string log = shader.getInfoLog();
        const char* dbg = shader.getInfoDebugLog();
        if (dbg && *dbg) { log += '\n'; log += dbg; }
        throw std::runtime_error("Shader parse error (" + path + "):\n" + log);
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(EShMsgDefault)) {
        std::string log = program.getInfoLog();
        const char* dbg = program.getInfoDebugLog();
        if (dbg && *dbg) { log += '\n'; log += dbg; }
        throw std::runtime_error("Shader link error (" + path + "):\n" + log);
    }

    glslang::TIntermediate* intermediate = program.getIntermediate(lang);
    if (!intermediate)
        throw std::runtime_error("No intermediate for shader: " + path);

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*intermediate, spirv);

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode    = spirv.data();

    VkShaderModule mod;
    VkResult res = vkCreateShaderModule(device_, &smci, nullptr, &mod);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module: " + path);

    return mod;
}

// ── pipeline ─────────────────────────────────────────────────────────────

void VulkanApp::createPipeline() {
    VkShaderModule vertMod = compileShader("shaders/fullscreen.vert",
                                           VK_SHADER_STAGE_VERTEX_BIT);
    VkShaderModule fragMod = compileShader("shaders/ray_march.frag",
                                           VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(swapchainExtent_.width);
    viewport.height   = static_cast<float>(swapchainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = swapchainExtent_;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.pViewports    = &viewport;
    vp.scissorCount  = 1;
    vp.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbAtt;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &descSetLayout_;
    VkResult res = vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create pipeline layout");

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pColorBlendState    = &cb;
    gpci.layout              = pipelineLayout_;
    gpci.renderPass          = renderPass_;

    res = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create graphics pipeline");

    vkDestroyShaderModule(device_, fragMod, nullptr);
    vkDestroyShaderModule(device_, vertMod, nullptr);
}

// ── framebuffers ─────────────────────────────────────────────────────────

void VulkanApp::createFramebuffers() {
    for (auto& f : frames_) {
        VkFramebufferCreateInfo fci{};
        fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass      = renderPass_;
        fci.attachmentCount = 1;
        fci.pAttachments    = &f.imageView;
        fci.width           = swapchainExtent_.width;
        fci.height          = swapchainExtent_.height;
        fci.layers          = 1;
        VkResult res = vkCreateFramebuffer(device_, &fci, nullptr, &f.framebuffer);
        if (res != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
    }
}

// ── command pool ─────────────────────────────────────────────────────────

void VulkanApp::createCommandPool() {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = graphicsQ_;
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult res = vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
}

void VulkanApp::createCommandBuffers() {
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = cmdPool_;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = static_cast<uint32_t>(frames_.size());

    std::vector<VkCommandBuffer> bufs(frames_.size());
    VkResult res = vkAllocateCommandBuffers(device_, &cbai, bufs.data());
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to allocate command buffers");

    for (size_t i = 0; i < frames_.size(); ++i)
        frames_[i].cmdBuffer = bufs[i];
}

// ── sync objects ─────────────────────────────────────────────────────────

void VulkanApp::createSyncObjects() {
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (auto& f : frames_) {
        VkResult r1 = vkCreateSemaphore(device_, &sci, nullptr, &f.imageAvailable);
        VkResult r2 = vkCreateSemaphore(device_, &sci, nullptr, &f.renderFinished);
        VkResult r3 = vkCreateFence(device_, &fci, nullptr, &f.inFlight);
        if (r1 != VK_SUCCESS || r2 != VK_SUCCESS || r3 != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects");
    }
}

// ── rebuild swapchain ────────────────────────────────────────────────────

void VulkanApp::rebuildSwapchain() {
    int w = window_->getWidth();
    int h = window_->getHeight();
    while (w == 0 || h == 0) {
        window_->pollEvents();
        w = window_->getWidth();
        h = window_->getHeight();
    }

    vkDeviceWaitIdle(device_);

    for (auto& f : frames_) {
        vkDestroyFramebuffer(device_, f.framebuffer, nullptr);
        vkDestroyImageView(device_, f.imageView, nullptr);
        vkDestroySemaphore(device_, f.imageAvailable, nullptr);
        vkDestroySemaphore(device_, f.renderFinished, nullptr);
        vkDestroyFence(device_, f.inFlight, nullptr);
    }
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);

    frames_.clear();
    createSwapchain();
    createFramebuffers();
    createSyncObjects();

    width_  = w;
    height_ = h;
}

// ── draw frame ───────────────────────────────────────────────────────────

void VulkanApp::drawFrame() {
    auto& frame = frames_[currentFrame_];

    vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
        frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        rebuildSwapchain();
        return;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    vkResetFences(device_, 1, &frame.inFlight);
    vkResetCommandBuffer(frame.cmdBuffer, 0);

    recordCommandBuffer(frame, imageIndex);
    updateUniforms();

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &frame.imageAvailable;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &frame.cmdBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &frame.renderFinished;

    VkResult res = vkQueueSubmit(graphicsQueue_, 1, &si, frame.inFlight);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to submit queue");

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &frame.renderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(presentQueue_, &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        rebuildSwapchain();
    }

    currentFrame_ = (currentFrame_ + 1) % static_cast<uint32_t>(frames_.size());
}

void VulkanApp::recordCommandBuffer(Frame& frame, uint32_t imageIndex) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(frame.cmdBuffer, &bi);

    VkClearValue clear;
    clear.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass        = renderPass_;
    rpbi.framebuffer       = frames_[imageIndex].framebuffer;
    rpbi.renderArea.extent = swapchainExtent_;
    rpbi.clearValueCount   = 1;
    rpbi.pClearValues      = &clear;
    vkCmdBeginRenderPass(frame.cmdBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(frame.cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(frame.cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1, &descSet_, 0, nullptr);
    vkCmdDraw(frame.cmdBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(frame.cmdBuffer);
    vkEndCommandBuffer(frame.cmdBuffer);
}

// ── update uniforms ──────────────────────────────────────────────────────

void VulkanApp::updateUniforms() {
    auto now = std::chrono::steady_clock::now();
    float t = std::chrono::duration<float>(now - startTime_).count();

    // Input-driven camera (orbit with drag, zoom with scroll)
    MouseState m = window_->consumeMouse();
    if (m.leftDown) {
        camTheta_ -= m.dx * 0.005f;
        camPhi_   += m.dy * 0.005f;
        camPhi_ = std::max(-1.5f, std::min(1.5f, camPhi_));
    }
    camDist_ *= (1.0f + m.scroll * 0.05f);
    camDist_ = std::max(1.0f, std::min(50.0f, camDist_));

    float cx = camDist_ * std::cos(camPhi_) * std::sin(camTheta_);
    float cy = camDist_ * std::sin(camPhi_);
    float cz = camDist_ * std::cos(camPhi_) * std::cos(camTheta_);

    UBO ubo{};
    ubo.time        = t;
    ubo.cameraPos   = {cx + camTarget_[0], cy + camTarget_[1], cz + camTarget_[2], 0.0f};
    ubo.cameraTarget= {camTarget_[0], camTarget_[1], camTarget_[2], 0.0f};

    void* data;
    vkMapMemory(device_, uboMem_, 0, sizeof(UBO), 0, &data);
    std::memcpy(data, &ubo, sizeof(UBO));
    vkUnmapMemory(device_, uboMem_);
}

// ── utility ──────────────────────────────────────────────────────────────

uint32_t VulkanApp::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((filter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

// ── cleanup ──────────────────────────────────────────────────────────────

void VulkanApp::cleanup() {
    for (auto& f : frames_) {
        vkDestroyFence(device_, f.inFlight, nullptr);
        vkDestroySemaphore(device_, f.renderFinished, nullptr);
        vkDestroySemaphore(device_, f.imageAvailable, nullptr);
        vkDestroyFramebuffer(device_, f.framebuffer, nullptr);
        vkDestroyImageView(device_, f.imageView, nullptr);
    }
    vkDestroyBuffer(device_, ubo_, nullptr);
    vkFreeMemory(device_, uboMem_, nullptr);
    vkDestroyDescriptorPool(device_, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
    vkDestroyPipeline(device_, pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    vkDestroyCommandPool(device_, cmdPool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (dbg_) vkDestroyDebugUtilsMessengerEXT(instance_, dbg_, nullptr);
    vkDestroyInstance(instance_, nullptr);
    glslang::FinalizeProcess();
    delete window_;
}
