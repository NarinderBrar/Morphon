#version 460

layout (location = 0) out vec2 uv;

void main() {
    uint idx = gl_VertexIndex;
    float x = float(int(idx & 1u) << 2) - 1.0f;
    float y = float(int(idx & 2u) << 1) - 1.0f;
    uv = vec2(x, -y) * 0.5f + 0.5f;
    gl_Position = vec4(x, y, 0.0f, 1.0f);
}
