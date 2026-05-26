#pragma once

#define None  0L
#define Bool  int

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

#undef None
#undef Bool

#include <volk.h>

#include <string>
#include <vector>

struct MouseState {
    int  x = 0, y = 0;
    int  dx = 0, dy = 0;
    bool leftDown = false;
    int  scroll = 0;
};

class X11Window {
public:
    X11Window(const char* title, int width, int height);
    ~X11Window();

    bool shouldClose() const { return closed_; }
    void pollEvents();
    int getWidth()  const { return width_; }
    int getHeight() const { return height_; }

    const MouseState& getMouse() const { return mouse_; }
    MouseState consumeMouse();

    VkSurfaceKHR createVulkanSurface(VkInstance instance);
    static std::vector<const char*> getVulkanExtensions();

private:
    Display* display_ = nullptr;
    Window   window_  = 0;
    Atom     wmDelete_ = 0;
    int      width_  = 0;
    int      height_ = 0;
    bool     closed_ = false;

    MouseState mouse_;
};
