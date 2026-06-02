#version 460

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

layout (binding = 0) uniform UBO {
    float time;
    float mouseNdcX;
    float mouseNdcY;
    int   editorMode;
    vec3  cameraPos;
    float mergeThreshold;
    vec3  cameraTarget;
    float _pad1;
    vec3  ghostPos;
    float ghostValid;
    vec4  ghostPrimInfo;
    vec3  selectedPos;
    float selectedValid;
    vec4  selectedPrimInfo;
    vec3  camRight;
    float _pad2;
    vec3  camUp;
    float _pad3;
    uvec4 hiddenFlags0;
    uvec4 hiddenFlags1;
    int   selectedCount;
    vec4  selPos[32];
    vec4  selInfo[32];
    mat4  viewProj;
} ubo;

layout (push_constant) uniform PushConstants {
    vec4  color;
    mat4  model;
} pc;

layout (location = 0) out vec3 fragColor;
layout (location = 1) out vec3 fragNormal;

void main() {
    gl_Position = ubo.viewProj * pc.model * vec4(aPos, 1.0);
    fragColor = pc.color.rgb;
    fragNormal = normalize(mat3(pc.model) * aNormal);
}
