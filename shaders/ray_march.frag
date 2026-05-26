#version 460

layout (location = 0) out vec4 outColor;
layout (location = 0) in vec2 uv;

layout (binding = 0) uniform UBO {
    float time;
    vec3  cameraPos;
    float _pad0;
    vec3  cameraTarget;
    float _pad1;
} ubo;

// ── Basic SDF primitives ─────────────────────────────────────────────────

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdPlane(vec3 p, float h) {
    return p.y - h;
}

float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

// ── Operations ───────────────────────────────────────────────────────────

float opUnion(float a, float b)  { return min(a, b); }
float opSubtract(float a, float b) { return max(a, -b); }
float opIntersect(float a, float b) { return max(a, b); }

float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float opSmoothSubtract(float a, float b, float k) {
    float h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}

vec3 opRepeat(vec3 p, vec3 c) {
    return p - c * round(p / c);
}

// ── Scene ────────────────────────────────────────────────────────────────

float scene(vec3 p) {
    float t = ubo.time;

    // Animated bouncing box
    vec3 bp = p;
    bp.x += 1.8f;
    float bounce = abs(sin(t * 0.8f)) * 0.8f + 0.2f;
    bp.y -= bounce - 0.3f;
    float box = sdBox(bp, vec3(0.5f, 0.5f, 0.5f));

    // Rotating torus
    vec3 tp = p;
    tp.x -= 1.8f;
    float ang = t * 0.5f;
    float ca = cos(ang), sa = sin(ang);
    tp = vec3(ca * tp.x + sa * tp.z, tp.y, -sa * tp.x + ca * tp.z);
    float torus = sdTorus(tp, vec2(0.7f, 0.3f));

    // Ground plane
    float ground = sdPlane(p, -1.5f);

    // Central sphere with slight wobble
    vec3 sp = p;
    sp.y += sin(length(sp.xz) * 2.0f + t * 1.5f) * 0.08f;
    float sphere = sdSphere(sp, 0.8f);

    // Animated floating octahedrons (array via repeat)
    vec3 op = p;
    op.y += 2.5f;
    op = opRepeat(op, vec3(2.5f, 0.0f, 2.5f));
    op.y += sin(t * 0.7f + op.x * 2.0f + op.z * 1.3f) * 0.2f;
    float oct = sdOctahedron(op, 0.25f);

    // Combine everything
    float d = opUnion(box, torus);
    d = opUnion(d, sphere);
    d = opUnion(d, ground);
    d = opSmoothUnion(d, oct, 0.1f);

    return d;
}

// ── Ray marching ─────────────────────────────────────────────────────────

struct MarchResult {
    float dist;
    int   steps;
    vec3  pos;
};

MarchResult march(vec3 ro, vec3 rd) {
    float t = 0.0f;
    int steps = 0;
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * t;
        float d = scene(p);
        if (d < 0.001f || t > 50.0f) break;
        t += d;
        steps = i;
    }
    return MarchResult(t, steps, ro + rd * t);
}

// ── Normal via numerical gradient ────────────────────────────────────────

vec3 normal(vec3 p) {
    const float e = 0.001f;
    vec2 eps = vec2(e, 0.0f);
    return normalize(vec3(
        scene(p + eps.xyy) - scene(p - eps.xyy),
        scene(p + eps.yxy) - scene(p - eps.yxy),
        scene(p + eps.yyx) - scene(p - eps.yyx)
    ));
}

// ── Soft shadow ──────────────────────────────────────────────────────────

float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0f;
    float t = mint;
    for (int i = 0; i < 48; i++) {
        float d = scene(ro + rd * t);
        res = min(res, k * d / t);
        if (res < 0.001f) break;
        t += d;
        if (t > maxt) break;
    }
    return clamp(res, 0.0f, 1.0f);
}

// ── Ambient occlusion ───────────────────────────────────────────────────

float ao(vec3 p, vec3 n) {
    float occ = 0.0f;
    float scale = 1.0f;
    for (int i = 0; i < 6; i++) {
        float dist = 0.02f + 0.04f * float(i);
        float d = scene(p + n * dist);
        occ += (dist - d) * scale;
        scale *= 0.85f;
    }
    return clamp(1.0f - occ * 2.0f, 0.0f, 1.0f);
}

// ── Main ─────────────────────────────────────────────────────────────────

void main() {
    // Camera
    vec3 ro = ubo.cameraPos;
    vec3 ta = ubo.cameraTarget;
    vec3 fwd = normalize(ta - ro);
    vec3 worldUp = vec3(0.0f, 1.0f, 0.0f);
    vec3 right = normalize(cross(fwd, worldUp));
    vec3 up = cross(right, fwd);

    float aspect = 16.0f / 9.0f;
    vec2 ndc = uv * 2.0f - 1.0f;
    ndc.x *= aspect;

    float fov = 1.0f;
    vec3 rd = normalize(fwd + right * ndc.x * fov + up * ndc.y * fov);

    // March
    MarchResult res = march(ro, rd);

    // Sky / background
    vec3 col = mix(
        vec3(0.05f, 0.05f, 0.10f),
        vec3(0.15f, 0.15f, 0.30f),
        max(rd.y, 0.0f)
    );

    if (res.dist < 50.0f) {
        vec3 p = res.pos;
        vec3 n = normal(p);

        // Lights
        vec3 lightDir1 = normalize(vec3(1.0f, 2.0f, 1.0f));
        vec3 lightDir2 = normalize(vec3(-1.0f, 0.5f, -1.0f));
        vec3 lightCol1 = vec3(1.0f, 0.9f, 0.7f);
        vec3 lightCol2 = vec3(0.5f, 0.6f, 1.0f);

        // Diffuse
        float diff1 = max(dot(n, lightDir1), 0.0f);
        float diff2 = max(dot(n, lightDir2), 0.0f);

        // Specular (Blinn-Phong)
        vec3 viewDir = normalize(ro - p);
        vec3 halfVec = normalize(lightDir1 + viewDir);
        float spec = pow(max(dot(n, halfVec), 0.0f), 64.0f);

        // Soft shadow
        float shadow = softShadow(p + n * 0.01f, lightDir1, 0.01f, 10.0f, 8.0f);

        // Ambient occlusion
        float ambient = ao(p, n);

        // Base colour by position
        vec3 baseColor = vec3(0.7f, 0.7f, 0.8f);
        {
            // Colour by gradient of position gives each shape a distinct tint
            baseColor = mix(
                vec3(0.8f, 0.2f, 0.3f),  // red (sphere area)
                vec3(0.1f, 0.7f, 0.8f),  // cyan (box area)
                smoothstep(-1.0f, 1.0f, p.x)
            );
            baseColor = mix(baseColor, vec3(1.0f, 0.7f, 0.1f),
                            smoothstep(0.0f, 0.5f, abs(p.y)) * 0.5f);
            // Ground colour
            if (p.y < -1.4f)
                baseColor = vec3(0.2f, 0.2f, 0.25f);
        }

        vec3 lighting = vec3(0.0f);
        lighting += lightCol1 * diff1 * shadow;
        lighting += lightCol2 * diff2 * 0.4f;
        lighting += vec3(0.02f); // ambient
        lighting *= ambient;

        col = baseColor * lighting + vec3(1.0f) * spec * shadow * 0.5f;

        // Fog
        float fog = 1.0f - exp(-res.dist * res.dist * 0.005f);
        col = mix(col, vec3(0.02f, 0.02f, 0.05f), fog);
    }

    // Vignette
    vec2 vig = (uv - 0.5f) * 1.2f;
    col *= 1.0f - dot(vig, vig) * 0.5f;

    // Reinhard tonemap + gamma
    col = col / (col + vec3(1.0f));
    col = pow(col, vec3(1.0f / 2.2f));

    outColor = vec4(col, 1.0f);
}
