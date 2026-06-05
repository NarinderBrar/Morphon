#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <volk.h>

#include <string>
#include <vector>

struct MouseState {
    int  x = 0, y = 0;
    int  dx = 0, dy = 0;
    bool leftDown = false;
    bool middleDown = false;
    int  scroll = 0;
};

class Win32Window {
public:
    Win32Window(const char* title, int width, int height);
    ~Win32Window();

    bool shouldClose() const { return closed_; }
    void pollEvents();
    int getWidth()  const { return width_; }
    int getHeight() const { return height_; }

    const MouseState& getMouse() const { return mouse_; }
    MouseState consumeMouse();
    bool isKeyDown(int vk) const { return keys_[vk & 0xFF]; }
    void setTitle(const char* title);

    VkSurfaceKHR createVulkanSurface(VkInstance instance);
    static std::vector<const char*> getVulkanExtensions();

    HWND getHwnd() const { return hwnd_; }

private:
    HWND   hwnd_ = nullptr;
    HINSTANCE hinstance_ = nullptr;
    int      width_  = 0;
    int      height_ = 0;
    bool     closed_ = false;

    MouseState mouse_;
    bool keys_[256]{};

    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
};
