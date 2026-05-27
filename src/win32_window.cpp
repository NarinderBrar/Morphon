#include "win32_window.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

Win32Window::Win32Window(const char* title, int width, int height)
    : width_(width), height_(height)
{
    hinstance_ = GetModuleHandleA(nullptr);

    const char* CLASS_NAME = "MorphonWindowClass";

    WNDCLASSA wc{};
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = wndProc;
    wc.hInstance     = hinstance_;
    wc.hCursor       = LoadCursorA(nullptr, IDC_ARROW);
    wc.lpszClassName = CLASS_NAME;

    RegisterClassA(&wc);

    RECT rect{0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    hwnd_ = CreateWindowExA(
        0,
        CLASS_NAME,
        title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        hinstance_,
        this
    );

    if (!hwnd_)
        throw std::runtime_error("Failed to create Win32 window");

    ShowWindow(hwnd_, SW_SHOW);
}

Win32Window::~Win32Window() {
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
}

void Win32Window::setTitle(const char* title) {
    SetWindowTextA(hwnd_, title);
}

MouseState Win32Window::consumeMouse() {
    MouseState m = mouse_;
    mouse_.dx = 0;
    mouse_.dy = 0;
    mouse_.scroll = 0;
    return m;
}

void Win32Window::pollEvents() {
    if (closed_) return;

    MSG msg{};
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
}

VkSurfaceKHR Win32Window::createVulkanSurface(VkInstance instance) {
    auto func = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
    if (!func)
        throw std::runtime_error("vkCreateWin32SurfaceKHR not available");

    VkWin32SurfaceCreateInfoKHR info{};
    info.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    info.hinstance = hinstance_;
    info.hwnd      = hwnd_;

    VkSurfaceKHR surface;
    VkResult res = func(instance, &info, nullptr, &surface);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create Win32 surface");

    return surface;
}

std::vector<const char*> Win32Window::getVulkanExtensions() {
    return {
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME
    };
}

LRESULT CALLBACK Win32Window::wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Win32Window* win = reinterpret_cast<Win32Window*>(
        GetWindowLongPtrA(hwnd, GWLP_USERDATA));

    if (msg == WM_CREATE) {
        CREATESTRUCTA* cs = reinterpret_cast<CREATESTRUCTA*>(lParam);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        win = reinterpret_cast<Win32Window*>(cs->lpCreateParams);
    }

    if (win) {
        switch (msg) {
        case WM_CLOSE:
            win->closed_ = true;
            return 0;
        case WM_SIZE:
            win->width_  = static_cast<int>(LOWORD(lParam));
            win->height_ = static_cast<int>(HIWORD(lParam));
            return 0;
        case WM_LBUTTONDOWN:
            SetCapture(hwnd);
            win->mouse_.leftDown = true;
            win->mouse_.x = static_cast<int>(LOWORD(lParam));
            win->mouse_.y = static_cast<int>(HIWORD(lParam));
            return 0;
        case WM_LBUTTONUP:
            ReleaseCapture();
            win->mouse_.leftDown = false;
            return 0;
        case WM_MOUSEMOVE:
            {
                int x = static_cast<int>(LOWORD(lParam));
                int y = static_cast<int>(HIWORD(lParam));
                win->mouse_.dx += x - win->mouse_.x;
                win->mouse_.dy += y - win->mouse_.y;
                win->mouse_.x = x;
                win->mouse_.y = y;
            }
            return 0;
        case WM_MOUSEWHEEL:
            win->mouse_.scroll += static_cast<int>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
            return 0;
        case WM_KEYDOWN:
            if (wParam < 256) win->keys_[wParam] = true;
            if (wParam == VK_ESCAPE || wParam == 'Q')
                win->closed_ = true;
            return 0;
        case WM_KEYUP:
            if (wParam < 256) win->keys_[wParam] = false;
            return 0;
        }
    }

    return DefWindowProcA(hwnd, msg, wParam, lParam);
}
