#version 460

layout (location = 0) in vec3 fragColor;
layout (location = 1) in vec3 fragNormal;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 nrm = normalize(gl_FrontFacing ? fragNormal : -fragNormal);
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.0));
    float diff = max(dot(nrm, lightDir), 0.0);
    float ambient = 0.35;
    vec3 col = fragColor * (ambient + diff * 0.65);
    outColor = vec4(col, 1.0);
}
