#version 460

layout (location = 0) out vec4 outColor;
layout (location = 0) in vec2 uv;

layout (binding = 0) uniform UBO {
    float time;
    float mouseNdcX;
    float mouseNdcY;
    int   editorMode;
    vec3  cameraPos;
    float mergeThreshold; // smooth-union blend radius (was _pad0)
    vec3  cameraTarget;
    float _pad1;
    vec3  ghostPos;
    float ghostValid;
    vec4  ghostPrimInfo; // x=primType, y=param1, z=param2, w=unused
    vec3  selectedPos;
    float selectedValid;
    vec4  selectedPrimInfo; // x=primType, y=param1, z=param2, w=showGizmo
    vec3  camRight;
    float _pad2;
    vec3  camUp;
    float _pad3;
    uint  hiddenFlags[8];

    // Multi-selection
    int   selectedCount;
    vec4  selPos[32];
    vec4  selInfo[32];
} ubo;

#define MAX_OBJECTS 256

struct PlacedObject {
    vec4 meta;  // x=type, y=primType, z=param1, w=param2
    vec4 pos;   // xyz=position, w=hidden
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

float sdCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCone(vec3 p, float h, float r) {
    vec2 q = vec2(length(p.xz), p.y);
    vec2 tip = vec2(r, -h);
    vec2 base = q - tip;
    float d = length(max(base, 0.0));
    float m = min(max(base.x, base.y), 0.0);
    return d + m;
}

float sdPlane(vec3 p, float h) { return p.y - h; }

// ── Operations ───────────────────────────────────────────────────────────

float opUnion(float a, float b) { return min(a, b); }
float opSubtract(float a, float b) { return max(a, -b); }
float opIntersect(float a, float b) { return max(a, b); }

float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 0.0001), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// ── Object evaluator ─────────────────────────────────────────────────────

float evalObject(vec3 p, PlacedObject obj) {
    int primType = int(obj.meta.y);
    float p1 = obj.meta.z;
    float p2 = obj.meta.w;
    vec3 op = p - obj.pos.xyz;

    float d;
    if (primType == 0) { // Box
        d = sdBox(op, vec3(p1));
    } else if (primType == 1) { // Sphere
        d = sdSphere(op, p1);
    } else if (primType == 2) { // Donut (torus)
        d = sdTorus(op, vec2(p1, p2));
    } else if (primType == 3) { // Cylinder
        d = sdCylinder(op, p1, p2);
    } else if (primType == 4) { // Pyramid (cone)
        d = sdCone(op, p1, p2);
    } else {
        d = sdBox(op, vec3(0.5));
    }
    return d;
}

// ── Scene ────────────────────────────────────────────────────────────────

float scene(vec3 p) {
    float ground = sdPlane(p, -1.5f);
    float d = ground;

    int objCount = min(objBuf.count, MAX_OBJECTS);
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (i >= objCount) break;

        // Skip hidden objects
        uint word = ubo.hiddenFlags[i >> 5];
        uint bit = (word >> (i & 31)) & 1u;
        if (bit != 0u) continue;

        float objDist = evalObject(p, objBuf.objects[i]);
        if (objBuf.objects[i].meta.x < 0.5f)
            d = opSmoothUnion(d, objDist, ubo.mergeThreshold);
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
        if (d < 0.002f || t > 50.0f) break;
        t += d;
        steps = i;
    }
    return MarchResult(t, steps, ro + rd * t);
}

vec3 normal(vec3 p) {
    int nc = min(objBuf.count, MAX_OBJECTS);
    bool nearVoid = false;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (i >= nc) break;
        if (objBuf.objects[i].meta.x < 0.5f) continue;
        float od = evalObject(p, objBuf.objects[i]);
        if (od < 0.1f) { nearVoid = true; break; }
    }

    float e = nearVoid ? 0.008f : 0.001f;
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
    vec2 ndcEdge = ndc;
    ndc.x *= aspect;

    float fov = 1.0f;
    vec3 rd = normalize(fwd + right * ndc.x * fov + up * ndc.y * fov);

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

        bool onVoid = false;
        bool onWall = false;
        float eps = 0.05f;
        int nc = min(objBuf.count, MAX_OBJECTS);
        for (int i = 0; i < MAX_OBJECTS; i++) {
            if (i >= nc) break;
            float od = evalObject(p, objBuf.objects[i]);
            if (od < eps) {
                if (objBuf.objects[i].meta.x > 0.5f)
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
            vec3 voidCol = vec3(0.0f, 0.8f, 0.9f) * lighting * 0.5f + vec3(1.0f) * spec * shadow * 0.25f;
            col = voidCol;
        } else if (onWall) {
            vec3 wallCol = vec3(0.0f, 0.8f, 0.9f) * lighting + vec3(1.0f) * spec * shadow * 0.5f;
            col = wallCol;
        } else {
            vec3 baseCol = baseColor * lighting + vec3(1.0f) * spec * shadow * 0.5f;
            col = baseCol;
        }

        float fog = 1.0f - exp(-res.dist * res.dist * 0.005f);
        col = mix(col, vec3(0.02f, 0.02f, 0.05f), fog);
    }

    // ── Shape-aware ghost preview ──────────────────────────────
    if (ubo.editorMode != 0 && ubo.ghostValid > 0.5f) {
        vec3 gp = ubo.ghostPos;
        float gType = ubo.ghostPrimInfo.x;
        float gP1 = ubo.ghostPrimInfo.y;
        float gP2 = ubo.ghostPrimInfo.z;

        vec3 ghostCol = (ubo.editorMode == 1) ? vec3(0.0, 1.0, 1.0) : vec3(1.0, 0.2, 0.2);
        float lineW = 0.006f;
        float bestDist = 1e10f;

        #define EDGE_SEG(wa, wb) { \
            vec3 _d_sa_ = (wa) - ro; \
            float _vz_sa_ = dot(_d_sa_, fwd); \
            vec2 _sa_ = _vz_sa_ <= 0.001 ? vec2(1e10) : (vec2(dot(_d_sa_, right), dot(_d_sa_, up)) / (_vz_sa_ * fov) / vec2(aspect, 1)); \
            vec3 _d_sb_ = (wb) - ro; \
            float _vz_sb_ = dot(_d_sb_, fwd); \
            vec2 _sb_ = _vz_sb_ <= 0.001 ? vec2(1e10) : (vec2(dot(_d_sb_, right), dot(_d_sb_, up)) / (_vz_sb_ * fov) / vec2(aspect, 1)); \
            if (_sa_.x < 1e9 && _sb_.x < 1e9) { \
                vec2 _ab_ = _sb_ - _sa_; \
                vec2 _ap_ = ndcEdge - _sa_; \
                float _t_ = clamp(dot(_ap_, _ab_) / dot(_ab_, _ab_), 0.0, 1.0); \
                bestDist = min(bestDist, length(ndcEdge - (_sa_ + _t_ * _ab_))); \
            } \
        }

        #define RING(segs, body) \
            for (int _ri_ = 0; _ri_ < (segs); _ri_++) { \
                float _a1_ = 6.283185 * float(_ri_) / float(segs); \
                float _a2_ = 6.283185 * float(_ri_ + 1) / float(segs); \
                { body } \
            }

        int giType = int(gType);
        if (giType == 0) { // Box
            vec3 he = vec3(gP1);
            vec3 c[8];
            for (int i = 0; i < 8; i++) {
                float bx = float(i & 1) * 2.0 - 1.0;
                float by = float((i >> 1) & 1) * 2.0 - 1.0;
                float bz = float((i >> 2) & 1) * 2.0 - 1.0;
                c[i] = gp + he * vec3(bx, by, bz);
            }
            EDGE_SEG(c[0], c[1]); EDGE_SEG(c[1], c[3]);
            EDGE_SEG(c[3], c[2]); EDGE_SEG(c[2], c[0]);
            EDGE_SEG(c[4], c[5]); EDGE_SEG(c[5], c[7]);
            EDGE_SEG(c[7], c[6]); EDGE_SEG(c[6], c[4]);
            EDGE_SEG(c[0], c[4]); EDGE_SEG(c[1], c[5]);
            EDGE_SEG(c[2], c[6]); EDGE_SEG(c[3], c[7]);
        } else if (giType == 1) { // Sphere
            float r = gP1;
            RING(20, { vec3 p1 = vec3(cos(_a1_)*r, sin(_a1_)*r, 0); vec3 p2 = vec3(cos(_a2_)*r, sin(_a2_)*r, 0); EDGE_SEG(gp+p1, gp+p2); })
            RING(20, { vec3 p1 = vec3(cos(_a1_)*r, 0, sin(_a1_)*r); vec3 p2 = vec3(cos(_a2_)*r, 0, sin(_a2_)*r); EDGE_SEG(gp+p1, gp+p2); })
            RING(20, { vec3 p1 = vec3(0, cos(_a1_)*r, sin(_a1_)*r); vec3 p2 = vec3(0, cos(_a2_)*r, sin(_a2_)*r); EDGE_SEG(gp+p1, gp+p2); })
        } else if (giType == 2) { // Donut
            float R = gP1, r = gP2;
            RING(24, { vec3 p1 = vec3(cos(_a1_)*R, 0, sin(_a1_)*R); vec3 p2 = vec3(cos(_a2_)*R, 0, sin(_a2_)*R); EDGE_SEG(gp+p1, gp+p2); })
            for (int ri = 0; ri < 4; ri++) {
                float angle = 6.283185 * float(ri) / 4.0;
                vec3 dir = vec3(cos(angle), 0, sin(angle));
                vec3 center = dir * R;
                vec3 upDir = vec3(0, 1, 0); vec3 radDir = dir;
                RING(12, { vec3 p1 = center + r * (cos(_a1_)*radDir + sin(_a1_)*upDir); vec3 p2 = center + r * (cos(_a2_)*radDir + sin(_a2_)*upDir); EDGE_SEG(gp+p1, gp+p2); })
            }
        } else if (giType == 3) { // Cylinder
            float hh = gP1, rr = gP2;
            RING(20, { vec3 p1 = vec3(cos(_a1_)*rr, hh, sin(_a1_)*rr); vec3 p2 = vec3(cos(_a2_)*rr, hh, sin(_a2_)*rr); EDGE_SEG(gp+p1, gp+p2); })
            RING(20, { vec3 p1 = vec3(cos(_a1_)*rr, -hh, sin(_a1_)*rr); vec3 p2 = vec3(cos(_a2_)*rr, -hh, sin(_a2_)*rr); EDGE_SEG(gp+p1, gp+p2); })
            for (int i = 0; i < 4; i++) { float a = 6.283185 * float(i) / 4.0; vec3 base = vec3(cos(a)*rr, -hh, sin(a)*rr); vec3 top = base + vec3(0, 2*hh, 0); EDGE_SEG(gp+base, gp+top); }
        } else if (giType == 4) { // Pyramid (cone)
            float hh = gP1, rr = gP2;
            RING(16, { vec3 p1 = vec3(cos(_a1_)*rr, -hh, sin(_a1_)*rr); vec3 p2 = vec3(cos(_a2_)*rr, -hh, sin(_a2_)*rr); EDGE_SEG(gp+p1, gp+p2); })
            vec3 tipOff = vec3(0, hh, 0);
            for (int i = 0; i < 16; i += 4) { float a = 6.283185 * float(i) / 16.0; vec3 base = vec3(cos(a)*rr, -hh, sin(a)*rr); EDGE_SEG(gp+base, gp+tipOff); }
        }

        #undef RING
        #undef EDGE_SEG
        #undef PROJ

        float lineAlpha = 1.0 - smoothstep(0.0, lineW, bestDist);
        float glowAlpha = 1.0 - smoothstep(0.0, lineW * 5.0, bestDist);
        col = mix(col, ghostCol, lineAlpha * 0.9);
        col += ghostCol * glowAlpha * 0.15;
    }

    // ── Shape-aware outline (green) for ALL selected objects ──
    {
        vec3 outColor = vec3(0.15, 0.95, 0.15);
        float lineW = 0.004f;
        float glowW = lineW * 4.0f;

        #define EDGE_SEG(wa, wb) { \
            vec3 _d_sa_ = (wa) - ro; \
            float _vz_sa_ = dot(_d_sa_, fwd); \
            vec2 _sa_ = _vz_sa_ <= 0.001 ? vec2(1e10) : (vec2(dot(_d_sa_, right), dot(_d_sa_, up)) / (_vz_sa_ * fov) / vec2(aspect, 1)); \
            vec3 _d_sb_ = (wb) - ro; \
            float _vz_sb_ = dot(_d_sb_, fwd); \
            vec2 _sb_ = _vz_sb_ <= 0.001 ? vec2(1e10) : (vec2(dot(_d_sb_, right), dot(_d_sb_, up)) / (_vz_sb_ * fov) / vec2(aspect, 1)); \
            if (_sa_.x < 1e9 && _sb_.x < 1e9) { \
                vec2 _ab_ = _sb_ - _sa_; \
                vec2 _ap_ = ndcEdge - _sa_; \
                float _t_ = clamp(dot(_ap_, _ab_) / dot(_ab_, _ab_), 0.0, 1.0); \
                bestDist = min(bestDist, length(ndcEdge - (_sa_ + _t_ * _ab_))); \
            } \
        }

        #define RING(segs, body) \
            for (int _ri_ = 0; _ri_ < (segs); _ri_++) { \
                float _a1_ = 6.283185 * float(_ri_) / float(segs); \
                float _a2_ = 6.283185 * float(_ri_ + 1) / float(segs); \
                { body } \
            }

        #define DRAW_SELECTED(gc, pType, pa1, pa2) \
            { \
                int _iType_ = int(pType); \
                if (_iType_ == 0) { \
                    vec3 he = vec3(pa1); \
                    vec3 c[8]; \
                    for (int i = 0; i < 8; i++) { \
                        float bx = float(i & 1) * 2.0 - 1.0; \
                        float by = float((i >> 1) & 1) * 2.0 - 1.0; \
                        float bz = float((i >> 2) & 1) * 2.0 - 1.0; \
                        c[i] = gc + he * vec3(bx, by, bz); \
                    } \
                    EDGE_SEG(c[0], c[1]); EDGE_SEG(c[1], c[3]); \
                    EDGE_SEG(c[3], c[2]); EDGE_SEG(c[2], c[0]); \
                    EDGE_SEG(c[4], c[5]); EDGE_SEG(c[5], c[7]); \
                    EDGE_SEG(c[7], c[6]); EDGE_SEG(c[6], c[4]); \
                    EDGE_SEG(c[0], c[4]); EDGE_SEG(c[1], c[5]); \
                    EDGE_SEG(c[2], c[6]); EDGE_SEG(c[3], c[7]); \
                } else if (_iType_ == 1) { \
                    float _r_ = pa1; \
                    RING(20, { vec3 p1 = vec3(cos(_a1_)*_r_, sin(_a1_)*_r_, 0); vec3 p2 = vec3(cos(_a2_)*_r_, sin(_a2_)*_r_, 0); EDGE_SEG(gc+p1, gc+p2); }) \
                    RING(20, { vec3 p1 = vec3(cos(_a1_)*_r_, 0, sin(_a1_)*_r_); vec3 p2 = vec3(cos(_a2_)*_r_, 0, sin(_a2_)*_r_); EDGE_SEG(gc+p1, gc+p2); }) \
                    RING(20, { vec3 p1 = vec3(0, cos(_a1_)*_r_, sin(_a1_)*_r_); vec3 p2 = vec3(0, cos(_a2_)*_r_, sin(_a2_)*_r_); EDGE_SEG(gc+p1, gc+p2); }) \
                } else if (_iType_ == 2) { \
                    float _R_ = pa1, _r_ = pa2; \
                    RING(24, { vec3 p1 = vec3(cos(_a1_)*_R_, 0, sin(_a1_)*_R_); vec3 p2 = vec3(cos(_a2_)*_R_, 0, sin(_a2_)*_R_); EDGE_SEG(gc+p1, gc+p2); }) \
                    for (int _ri_ = 0; _ri_ < 4; _ri_++) { \
                        float _angle_ = 6.283185 * float(_ri_) / 4.0; \
                        vec3 _dir_ = vec3(cos(_angle_), 0, sin(_angle_)); \
                        vec3 _center_ = _dir_ * _R_; \
                        vec3 _upDir_ = vec3(0, 1, 0); vec3 _radDir_ = _dir_; \
                        RING(12, { vec3 _p1_ = _center_ + _r_ * (cos(_a1_)*_radDir_ + sin(_a1_)*_upDir_); vec3 _p2_ = _center_ + _r_ * (cos(_a2_)*_radDir_ + sin(_a2_)*_upDir_); EDGE_SEG(gc+_p1_, gc+_p2_); }) \
                    } \
                } else if (_iType_ == 3) { \
                    float _hh_ = pa1, _rr_ = pa2; \
                    RING(20, { vec3 p1 = vec3(cos(_a1_)*_rr_, _hh_, sin(_a1_)*_rr_); vec3 p2 = vec3(cos(_a2_)*_rr_, _hh_, sin(_a2_)*_rr_); EDGE_SEG(gc+p1, gc+p2); }) \
                    RING(20, { vec3 p1 = vec3(cos(_a1_)*_rr_, -_hh_, sin(_a1_)*_rr_); vec3 p2 = vec3(cos(_a2_)*_rr_, -_hh_, sin(_a2_)*_rr_); EDGE_SEG(gc+p1, gc+p2); }) \
                    for (int _i_ = 0; _i_ < 4; _i_++) { float _a_ = 6.283185 * float(_i_) / 4.0; vec3 _base_ = vec3(cos(_a_)*_rr_, -_hh_, sin(_a_)*_rr_); vec3 _top_ = _base_ + vec3(0, 2*_hh_, 0); EDGE_SEG(gc+_base_, gc+_top_); } \
                } else if (_iType_ == 4) { \
                    float _hh_ = pa1, _rr_ = pa2; \
                    RING(16, { vec3 p1 = vec3(cos(_a1_)*_rr_, -_hh_, sin(_a1_)*_rr_); vec3 p2 = vec3(cos(_a2_)*_rr_, -_hh_, sin(_a2_)*_rr_); EDGE_SEG(gc+p1, gc+p2); }) \
                    vec3 _tipOff_ = vec3(0, _hh_, 0); \
                    for (int _i_ = 0; _i_ < 16; _i_ += 4) { float _a_ = 6.283185 * float(_i_) / 16.0; vec3 _base_ = vec3(cos(_a_)*_rr_, -_hh_, sin(_a_)*_rr_); EDGE_SEG(gc+_base_, gc+_tipOff_); } \
                } \
            }

        // Outline for primary selection
        float bestDist = 1e10f;
        if (ubo.selectedValid > 0.5f) {
            DRAW_SELECTED(ubo.selectedPos, ubo.selectedPrimInfo.x, ubo.selectedPrimInfo.y, ubo.selectedPrimInfo.z);
        }

        // Outline for additional multi-selected objects
        for (int _si_ = 0; _si_ < 32; _si_++) {
            if (ubo.selPos[_si_].w > 0.5f) {
                DRAW_SELECTED(ubo.selPos[_si_].xyz, ubo.selInfo[_si_].x, ubo.selInfo[_si_].y, ubo.selInfo[_si_].z);
            }
        }

        #undef DRAW_SELECTED
        #undef RING
        #undef EDGE_SEG

        float lineAlpha = 1.0 - smoothstep(0.0, lineW, bestDist);
        float glowAlpha = 1.0 - smoothstep(0.0, glowW, bestDist);
        col = mix(col, outColor, lineAlpha * 0.9);
        col += outColor * glowAlpha * 0.15;
    }

    // ── Gizmo cones (Move tool only, primary selection) ─────
    if (ubo.selectedPrimInfo.w > 0.5f && ubo.selectedValid > 0.5f) {
        vec3 gc = ubo.selectedPos;
        float gLen = 0.6f;
        float coneH = 0.15f;
        float coneR = 0.06f;
        float edgeW = 0.005f;
        vec3 gAxes[3] = { vec3(1,0,0), vec3(0,1,0), vec3(0,0,1) };
        vec3 gCols[3] = { vec3(1,0.2,0.2), vec3(0.2,1,0.2), vec3(0.2,0.2,1) };
        vec3 lightDir = normalize(vec3(1.0, 2.0, 1.0));
        for (int ai = 0; ai < 3; ai++) {
            vec3 basePos = gc + gAxes[ai] * gLen;
            vec3 apexPos = gc + gAxes[ai] * (gLen + coneH);

            // ── Screen-space triangle for silhouette test ──
            vec3 db = basePos - ro;
            float vzb = dot(db, fwd);
            if (vzb <= 0.001) continue;
            vec2 sBase = vec2(dot(db, right), dot(db, up)) / (vzb * fov);
            sBase.x /= aspect;

            vec3 da = apexPos - ro;
            float vza = dot(da, fwd);
            if (vza <= 0.001) continue;
            vec2 sApex = vec2(dot(da, right), dot(da, up)) / (vza * fov);
            sApex.x /= aspect;

            vec2 dirNDC = sApex - sBase;
            float dirLen = length(dirNDC);
            if (dirLen < 0.001) continue;
            dirNDC /= dirLen;
            vec2 perp = vec2(-dirNDC.y, dirNDC.x);
            float baseW = coneR / (vzb * fov);

            vec2 v0 = sApex, v1 = sBase + perp * baseW, v2 = sBase - perp * baseW;
            float s0 = (v1.x - v0.x) * (ndcEdge.y - v0.y) - (v1.y - v0.y) * (ndcEdge.x - v0.x);
            float s1 = (v2.x - v1.x) * (ndcEdge.y - v1.y) - (v2.y - v1.y) * (ndcEdge.x - v1.x);
            float s2 = (v0.x - v2.x) * (ndcEdge.y - v2.y) - (v0.y - v2.y) * (ndcEdge.x - v2.x);
            bool insideTri = (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) ||
                             (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0);
            if (!insideTri) continue;

            float edgeDist = min(min(abs(s0) / length(v1 - v0),
                                     abs(s1) / length(v2 - v1)),
                                 abs(s2) / length(v0 - v2));

            // ── 3D ray-cone intersection ──
            vec3 localY = -gAxes[ai];
            vec3 localX = normalize(cross(abs(localY.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0), localY));
            vec3 localZ = cross(localY, localX);

            vec3 roL = vec3(dot(ro - apexPos, localX),
                            dot(ro - apexPos, localY),
                            dot(ro - apexPos, localZ));
            vec3 rdL = vec3(dot(rd, localX), dot(rd, localY), dot(rd, localZ));

            float k = coneR / coneH;
            float k2 = k * k;
            float A = rdL.x*rdL.x + rdL.z*rdL.z - k2*rdL.y*rdL.y;
            float B = 2.0*(roL.x*rdL.x + roL.z*rdL.z - k2*roL.y*rdL.y);
            float C = roL.x*roL.x + roL.z*roL.z - k2*roL.y*roL.y;
            float disc = B*B - 4.0*A*C;
            if (disc < 0.0) continue;

            float sqrtDisc = sqrt(disc);
            float tHit = 1e10;
            bool hitCap = false;

            for (int ti = 0; ti < 2; ti++) {
                float t = (ti == 0) ? (-B - sqrtDisc) / (2.0*A)
                                    : (-B + sqrtDisc) / (2.0*A);
                if (t > 0.001) {
                    vec3 pL = roL + rdL * t;
                    if (pL.y >= 0.0 && pL.y <= coneH) {
                        tHit = t;
                        break;
                    }
                }
            }

            if (tHit > 1e9 && abs(rdL.y) > 1e-8) {
                float tCap = (coneH - roL.y) / rdL.y;
                if (tCap > 0.001) {
                    vec3 pCap = roL + rdL * tCap;
                    if (pCap.x*pCap.x + pCap.z*pCap.z <= coneR*coneR) {
                        tHit = tCap;
                        hitCap = true;
                    }
                }
            }

            if (tHit > 1e9) continue;

            // ── Normal and lighting ──
            vec3 nL;
            if (hitCap) {
                nL = vec3(0.0, 1.0, 0.0);
            } else {
                vec3 pHit = roL + rdL * tHit;
                nL = normalize(vec3(pHit.x, -k2 * pHit.y, pHit.z));
            }

            vec3 norm = nL.x*localX + nL.y*localY + nL.z*localZ;
            float diff = max(dot(norm, lightDir), 0.0);
            float lum = 0.3 + diff * 0.7;

            vec3 coneCol = gCols[ai] * lum;
            float coneAlpha = smoothstep(-edgeW, edgeW, edgeDist);
            col = mix(col, coneCol, coneAlpha);
        }
    }

    // Vignette
    vec2 vig = (uv - 0.5f) * 1.2f;
    col *= 1.0f - dot(vig, vig) * 0.5f;

    col = col / (col + vec3(1.0f));
    col = pow(col, vec3(1.0f / 2.2f));

    outColor = vec4(col, 1.0f);
}
