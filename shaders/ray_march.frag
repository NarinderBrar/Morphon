#version 460

layout (location = 0) out vec4 outColor;
layout (location = 0) in vec2 uv;

layout (binding = 0) uniform UBO {
    float time;
    float mouseNdcX;
    float mouseNdcY;
    int   editorMode;
    vec3  cameraPos;
    float _pad0;
    vec3  cameraTarget;
    float _pad1;
    vec3  ghostPos;
    float ghostValid;
} ubo;

#define MAX_OBJECTS 256

struct PlacedObject {
    vec4 type_size;
    vec4 pos;
};

layout (binding = 1, std430) readonly buffer ObjectBuffer {
    int count;
    uint _pad0, _pad1, _pad2;
    PlacedObject objects[];
} objBuf;

// ── SDF primitives ───────────────────────────────────────────────────────

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdPlane(vec3 p, float h) { return p.y - h; }

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

// ── Operations ───────────────────────────────────────────────────────────

float opUnion(float a, float b) { return min(a, b); }
float opSubtract(float a, float b) { return max(a, -b); }

float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

vec3 opRepeat(vec3 p, vec3 c) {
    return p - c * round(p / c);
}

// ── Scene ────────────────────────────────────────────────────────────────

float scene(vec3 p) {
    float t = ubo.time;

    vec3 bp = p;
    bp.x += 1.8f;
    float bounce = abs(sin(t * 0.8f)) * 0.8f + 0.2f;
    bp.y -= bounce - 0.3f;
    float box = sdBox(bp, vec3(0.5f, 0.5f, 0.5f));

    vec3 tp = p;
    tp.x -= 1.8f;
    float ang = t * 0.5f;
    float ca = cos(ang), sa = sin(ang);
    tp = vec3(ca * tp.x + sa * tp.z, tp.y, -sa * tp.x + ca * tp.z);
    float torus = sdTorus(tp, vec2(0.7f, 0.3f));

    float ground = sdPlane(p, -1.5f);

    vec3 sp = p;
    sp.y += sin(length(sp.xz) * 2.0f + t * 1.5f) * 0.08f;
    float sphere = sdSphere(sp, 0.8f);

    vec3 op = p;
    op.y += 2.5f;
    op = opRepeat(op, vec3(2.5f, 0.0f, 2.5f));
    op.y += sin(t * 0.7f + op.x * 2.0f + op.z * 1.3f) * 0.2f;
    float oct = sdOctahedron(op, 0.25f);

    float d = opUnion(box, torus);
    d = opUnion(d, sphere);
    d = opUnion(d, ground);
    d = opSmoothUnion(d, oct, 0.1f);

    int objCount = min(objBuf.count, MAX_OBJECTS);
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (i >= objCount) break;
        vec3 h = objBuf.objects[i].type_size.yzw;
        vec3 cpos = objBuf.objects[i].pos.xyz;
        float objDist = sdBox(p - cpos, h);
        if (objBuf.objects[i].type_size.x < 0.5f)
            d = opUnion(d, objDist);
        else
            d = opSubtract(d, objDist);
    }

    return d;
}

// ── Ray marching ─────────────────────────────────────────────────────────

struct MarchResult { float dist; int steps; vec3 pos; };

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

vec3 normal(vec3 p) {
    const float e = 0.001f;
    vec2 eps = vec2(e, 0.0f);
    return normalize(vec3(
        scene(p + eps.xyy) - scene(p - eps.xyy),
        scene(p + eps.yxy) - scene(p - eps.yxy),
        scene(p + eps.yyx) - scene(p - eps.yyx)
    ));
}

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

// ── Ground grid ──────────────────────────────────────────────────────────

float gridPattern(vec2 p) {
    vec2 f = abs(fract(p) - 0.5f);
    float d = min(f.x, f.y);
    return smoothstep(0.03f, 0.0f, d);
}

// ── Main ─────────────────────────────────────────────────────────────────

void main() {
    vec3 ro = ubo.cameraPos;
    vec3 ta = ubo.cameraTarget;
    vec3 fwd = normalize(ta - ro);
    vec3 worldUp = vec3(0.0f, 1.0f, 0.0f);
    vec3 right = normalize(cross(fwd, worldUp));
    vec3 up = cross(right, fwd);

    float aspect = 16.0f / 9.0f;
    vec2 ndc = uv * 2.0f - 1.0f;
    vec2 ndcEdge = ndc; // non-aspect-corrected for wireframe edge test
    ndc.x *= aspect;

    float fov = 1.0f;
    vec3 rd = normalize(fwd + right * ndc.x * fov + up * ndc.y * fov);

    // Ghost preview center from CPU ray-march (passed via UBO)
    vec3 ghostCenter = ubo.ghostPos;

    MarchResult res = march(ro, rd);

    vec3 col = mix(
        vec3(0.05f, 0.05f, 0.10f),
        vec3(0.15f, 0.15f, 0.30f),
        max(rd.y, 0.0f)
    );

    if (res.dist < 50.0f) {
        vec3 p = res.pos;
        vec3 n = normal(p);

        vec3 lightDir1 = normalize(vec3(1.0f, 2.0f, 1.0f));
        vec3 lightDir2 = normalize(vec3(-1.0f, 0.5f, -1.0f));
        vec3 lightCol1 = vec3(1.0f, 0.9f, 0.7f);
        vec3 lightCol2 = vec3(0.5f, 0.6f, 1.0f);

        float diff1 = max(dot(n, lightDir1), 0.0f);
        float diff2 = max(dot(n, lightDir2), 0.0f);

        vec3 viewDir = normalize(ro - p);
        vec3 halfVec = normalize(lightDir1 + viewDir);
        float spec = pow(max(dot(n, halfVec), 0.0f), 64.0f);

        float shadow = softShadow(p + n * 0.01f, lightDir1, 0.01f, 10.0f, 8.0f);
        float ambient = ao(p, n);

        vec3 baseColor = vec3(0.7f, 0.7f, 0.8f);
        {
            baseColor = mix(
                vec3(0.8f, 0.2f, 0.3f),
                vec3(0.1f, 0.7f, 0.8f),
                smoothstep(-1.0f, 1.0f, p.x)
            );
            baseColor = mix(baseColor, vec3(1.0f, 0.7f, 0.1f),
                            smoothstep(0.0f, 0.5f, abs(p.y)) * 0.5f);
            if (p.y < -1.4f) {
                baseColor = vec3(0.2f, 0.2f, 0.25f);
                float grid = gridPattern(p.xz);
                baseColor += grid * vec3(0.15f, 0.15f, 0.20f);
            }
        }

        // Check if hit point is on a wall or void surface
        bool onVoid = false;
        bool onWall = false;
        float eps = 0.01f;
        int nc = min(objBuf.count, MAX_OBJECTS);
        for (int i = 0; i < MAX_OBJECTS; i++) {
            if (i >= nc) break;
            vec3 h = objBuf.objects[i].type_size.yzw;
            vec3 cpos = objBuf.objects[i].pos.xyz;
            float od = sdBox(p - cpos, h);
            if (od < eps) {
                if (objBuf.objects[i].type_size.x > 0.5f)
                    onVoid = true;
                else
                    onWall = true;
            }
        }

        vec3 lighting = vec3(0.0f);
        lighting += lightCol1 * diff1 * shadow;
        lighting += lightCol2 * diff2 * 0.4f;
        lighting += vec3(0.02f);
        lighting *= ambient;

        if (onVoid) {
            // Void: dark abyss - deep purple/black with faint glow
            vec3 voidCol = vec3(0.05f, 0.0f, 0.08f) * lighting;
            voidCol += vec3(0.03f, 0.0f, 0.05f);
            col = voidCol;
        } else if (onWall) {
            // Wall: cyan accent
            vec3 wallCol = vec3(0.0f, 0.8f, 0.9f) * lighting + vec3(1.0f) * spec * shadow * 0.5f;
            col = wallCol;
        } else {
            vec3 baseCol = baseColor * lighting + vec3(1.0f) * spec * shadow * 0.5f;
            col = baseCol;
        }

        float fog = 1.0f - exp(-res.dist * res.dist * 0.005f);
        col = mix(col, vec3(0.02f, 0.02f, 0.05f), fog);
    }

    // ── HUD bottom bar ───────────────────────────────────────────────────
    vec3 accentCol = vec3(0.3f, 0.5f, 0.8f);
    if (ubo.editorMode == 1) accentCol = vec3(0.0f, 0.8f, 0.8f);
    if (ubo.editorMode == 2) accentCol = vec3(0.8f, 0.2f, 0.2f);

    float hudY = uv.y;
    float hudFade = smoothstep(0.0f, 0.04f, hudY) * (1.0f - smoothstep(0.08f, 0.14f, hudY));
    if (hudFade > 0.0f) {
        col = mix(col, vec3(0.04f, 0.04f, 0.06f), hudFade * 0.85f);
    }

    // Accent line
    float accentLine = smoothstep(0.002f, 0.0f, abs(hudY - 0.08f));
    col = mix(col, accentCol, accentLine * 0.5f);

    // Mode indicator dots
    float dotR = 0.012f;
    vec2 dotBase = vec2(0.5f - 0.20f, 0.055f);
    for (int i = 0; i < 3; i++) {
        vec2 dc = uv - dotBase - vec2(float(i) * 0.08f, 0.0f);
        dc.x *= aspect;
        float dd = length(dc);
        float dotFill = 1.0f - smoothstep(dotR * 0.8f, dotR, dd);
        vec3 dotCol = vec3(0.15f);
        if (i == 0) dotCol = vec3(0.3f, 0.5f, 0.8f);
        if (i == 1) dotCol = vec3(0.0f, 0.8f, 0.8f);
        if (i == 2) dotCol = vec3(0.8f, 0.2f, 0.2f);
        if (i == ubo.editorMode) {
            // Glow ring around active
            float ring = 1.0f - smoothstep(dotR * 1.2f, dotR * 1.8f, dd);
            col = mix(col, dotCol * 1.5f, ring * 0.3f);
        }
        col = mix(col, dotCol, dotFill * 0.8f);
    }

    // ── Wireframe ghost preview ──────────────────────────────────────────
    if (ubo.editorMode != 0 && ubo.ghostValid > 0.5f) {
        vec3 hf = vec3(0.5f);
        vec3 corners[8];
        for (int i = 0; i < 8; i++) {
            float bx = float(i & 1) * 2.0f - 1.0f;
            float by = float((i >> 1) & 1) * 2.0f - 1.0f;
            float bz = float((i >> 2) & 1) * 2.0f - 1.0f;
            corners[i] = ghostCenter + hf * vec3(bx, by, bz);
        }

        vec3 sndc[8];
        bool allValid = true;
        for (int i = 0; i < 8; i++) {
            vec3 dir = corners[i] - ro;
            float vz = dot(dir, fwd);
            if (vz <= 0.001f) { allValid = false; break; }
            float vx = dot(dir, right);
            float vy = dot(dir, up);
            vec2 s = vec2(vx, vy) / (vz * fov);
            s.x /= aspect;
            sndc[i] = vec3(s, 1.0f);
        }

        if (allValid) {
            float lineW = 0.006f;
            vec2 pn = ndcEdge;
            float bestDist = 1e10f;

            // 12 edges of the box, unrolled
            #define EDGE(ai, bi) { vec2 a=sndc[ai].xy; vec2 b=sndc[bi].xy; vec2 ab=b-a; vec2 ap=pn-a; float t=clamp(dot(ap,ab)/dot(ab,ab),0.0,1.0); bestDist=min(bestDist,length(pn-(a+t*ab))); }
            EDGE(0,1); EDGE(1,3); EDGE(3,2); EDGE(2,0);
            EDGE(4,5); EDGE(5,7); EDGE(7,6); EDGE(6,4);
            EDGE(0,4); EDGE(1,5); EDGE(2,6); EDGE(3,7);
            #undef EDGE

            float lineAlpha = 1.0f - smoothstep(0.0f, lineW, bestDist);
            float glowAlpha = 1.0f - smoothstep(0.0f, lineW * 3.0f, bestDist);
            vec3 ghostCol = (ubo.editorMode == 1) ? vec3(0.0f, 1.0f, 1.0f) : vec3(1.0f, 0.2f, 0.2f);
            col = mix(col, ghostCol, lineAlpha * 0.9f);
            col += ghostCol * glowAlpha * 0.1f;
        }
    }

    // Vignette
    vec2 vig = (uv - 0.5f) * 1.2f;
    col *= 1.0f - dot(vig, vig) * 0.5f;

    col = col / (col + vec3(1.0f));
    col = pow(col, vec3(1.0f / 2.2f));

    outColor = vec4(col, 1.0f);
}
