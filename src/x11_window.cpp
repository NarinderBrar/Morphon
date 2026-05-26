#include "x11_window.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <dlfcn.h>

X11Window::X11Window(const char* title, int width, int height)
    : width_(width), height_(height)
{
    display_ = XOpenDisplay(nullptr);
    if (!display_)
        throw std::runtime_error("Cannot open X display");

    int screen = DefaultScreen(display_);
    Window root = RootWindow(display_, screen);

    window_ = XCreateSimpleWindow(display_, root,
        0, 0, width, height, 0,
        BlackPixel(display_, screen),
        WhitePixel(display_, screen));

    XStoreName(display_, window_, title);

    XSelectInput(display_, window_,
        ExposureMask | KeyPressMask | StructureNotifyMask |
        ButtonPressMask | ButtonReleaseMask |
        Button1MotionMask | PointerMotionMask);

    wmDelete_ = XInternAtom(display_, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(display_, window_, &wmDelete_, 1);

    XMapWindow(display_, window_);
    XFlush(display_);
}

X11Window::~X11Window() {
    if (display_) {
        if (window_) XDestroyWindow(display_, window_);
        XCloseDisplay(display_);
    }
}

MouseState X11Window::consumeMouse() {
    MouseState m = mouse_;
    mouse_.dx = 0;
    mouse_.dy = 0;
    mouse_.scroll = 0;
    return m;
}

void X11Window::pollEvents() {
    if (closed_) return;

    while (XPending(display_) > 0) {
        XEvent ev;
        XNextEvent(display_, &ev);

        switch (ev.type) {
        case ClientMessage:
            if (static_cast<Atom>(ev.xclient.data.l[0]) == wmDelete_)
                closed_ = true;
            break;
        case KeyPress: {
            KeySym ks = XLookupKeysym(&ev.xkey, 0);
            if (ks == XK_Escape || ks == XK_q || ks == XK_Q)
                closed_ = true;
            break;
        }
        case ConfigureNotify:
            width_  = ev.xconfigure.width;
            height_ = ev.xconfigure.height;
            break;
        case ButtonPress:
            if (ev.xbutton.button == Button1) {
                mouse_.leftDown = true;
            } else if (ev.xbutton.button == Button4) {
                mouse_.scroll += 1;
            } else if (ev.xbutton.button == Button5) {
                mouse_.scroll -= 1;
            }
            break;
        case ButtonRelease:
            if (ev.xbutton.button == Button1) {
                mouse_.leftDown = false;
            }
            break;
        case MotionNotify:
            mouse_.dx += ev.xmotion.x - mouse_.x;
            mouse_.dy += ev.xmotion.y - mouse_.y;
            mouse_.x = ev.xmotion.x;
            mouse_.y = ev.xmotion.y;
            break;
        default:
            break;
        }
    }
}

VkSurfaceKHR X11Window::createVulkanSurface(VkInstance instance) {
    auto func = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(
        vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR"));
    if (!func)
        throw std::runtime_error("vkCreateXlibSurfaceKHR not available");

    VkXlibSurfaceCreateInfoKHR info{};
    info.sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    info.dpy    = display_;
    info.window = window_;

    VkSurfaceKHR surface;
    VkResult res = func(instance, &info, nullptr, &surface);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create Xlib surface");

    return surface;
}

std::vector<const char*> X11Window::getVulkanExtensions() {
    return {
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME
    };
}
