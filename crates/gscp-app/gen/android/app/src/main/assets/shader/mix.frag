#version 100
#extension GL_OES_EGL_image_external : require
precision mediump float;

// 移植自桌面端 WGSL 合成管线（crates/gscp-player/src/render.rs）：
// overlay = 5 点 AA + 边缘锐化 → 黑边抠像(luma 阈值+羽化) → 饱和度增强 ×1.12
//           → 加色混合；底图在 overlay 圆角矩形区域内按 dim_strength 压暗。

varying vec2 vTexCoord;

uniform samplerExternalOES uBottomTexture;
uniform samplerExternalOES uTopTexture;
uniform mat4 uBottomViewMatrix;
uniform int uBottomMirror;
uniform float uBaseBrightness;

// overlay 几何（屏幕 uv 空间的轴对齐矩形）
uniform vec4 uTopRect;      // (centerX, centerY, halfW, halfH)
uniform int uTopRotation;   // 旋转象限 0..3（顺时针 90° 步进）
uniform int uTopMirror;
uniform int uTopEnable;
uniform vec2 uTopTexel;     // 1/overlay 纹理宽高

// overlay 效果参数（与桌面 EffectParams 同名同语义）
uniform float uOverlayAlpha;
uniform float uOverlayBrightness;
uniform float uOverlaySaturation;
uniform float uDimStrength;
uniform float uKeyLow;
uniform float uKeyHigh;
uniform float uFeatherPower;
uniform float uFeatherRadius;

vec2 rotQuad(vec2 t, int q) {
    if (q == 1) return vec2(1.0 - t.y, t.x);
    if (q == 2) return vec2(1.0 - t.x, 1.0 - t.y);
    if (q == 3) return vec2(t.y, 1.0 - t.x);
    return t;
}

vec3 boostSaturation(vec3 color, float gain) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 gray = vec3(luma);
    return clamp(gray + (color - gray) * gain, 0.0, 1.0);
}

// 黑边抠像：luma 低于 key_low → 全透明，高于 key_high → 不变，
// 中间按 feather_power 曲线羽化；feather_radius 外扩阈值区间。
float blackKeyAlpha(vec3 rgb, float srcAlpha) {
    if (uKeyHigh <= uKeyLow) return srcAlpha;
    float lo = max(uKeyLow - uFeatherRadius, 0.0);
    float hi = min(uKeyHigh + uFeatherRadius, 1.0);
    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
    float t = clamp((luma - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
    return srcAlpha * pow(t, uFeatherPower);
}

// 圆角 dim mask（overlay 本地 rect 空间，mask 覆盖中部 100%×60%，宽羽化）
float dimMask(vec2 local) {
    vec2 maskSize = vec2(1.0, 0.6);
    vec2 maskHalf = maskSize * 0.5;
    float maskRadius = maskSize.y * 0.08;
    float featherWidth = maskSize.x * 0.6;
    vec2 delta = abs(local - vec2(0.5)) - (maskHalf - vec2(maskRadius));
    float outside = length(max(delta, vec2(0.0)));
    float inside = min(max(delta.x, delta.y), 0.0);
    float signedDistance = outside + inside - maskRadius;
    return 1.0 - smoothstep(0.0, featherWidth, signedDistance);
}

// 5 点 AA + 边缘锐化（与桌面 sample_overlay_filtered 一致）
vec4 sampleTopFiltered(vec2 t) {
    vec2 aaOffset = uTopTexel * 0.65;
    vec4 center = texture2D(uTopTexture, t);
    vec4 aaColor = center * 0.4
        + texture2D(uTopTexture, t + vec2(-aaOffset.x, -aaOffset.y)) * 0.15
        + texture2D(uTopTexture, t + vec2( aaOffset.x, -aaOffset.y)) * 0.15
        + texture2D(uTopTexture, t + vec2(-aaOffset.x,  aaOffset.y)) * 0.15
        + texture2D(uTopTexture, t + vec2( aaOffset.x,  aaOffset.y)) * 0.15;
    vec4 neighbors = (
        texture2D(uTopTexture, t + vec2( uTopTexel.x, 0.0)) +
        texture2D(uTopTexture, t + vec2(-uTopTexel.x, 0.0)) +
        texture2D(uTopTexture, t + vec2(0.0,  uTopTexel.y)) +
        texture2D(uTopTexture, t + vec2(0.0, -uTopTexel.y))
    ) * 0.25;
    float edgeAlpha = abs(center.a - neighbors.a);
    float edgeLuma = abs(dot(center.rgb - neighbors.rgb, vec3(0.299, 0.587, 0.114)));
    float edgeMask = smoothstep(0.02, 0.12, max(edgeAlpha, edgeLuma * 1.5));
    vec3 sharpened = clamp(aaColor.rgb + (center.rgb - neighbors.rgb) * 0.8 * edgeMask, 0.0, 1.0);
    return vec4(sharpened, aaColor.a);
}

void main() {
    // 底图：沿用原矩阵路径（fill/contain + 旋转）
    vec4 bCoord4 = uBottomViewMatrix * vec4(vTexCoord, 0.0, 1.0);
    vec2 bCoord = bCoord4.xy;
    if (uBottomMirror == 1) bCoord.x = 1.0 - bCoord.x;
    vec3 color = texture2D(uBottomTexture, bCoord).rgb * uBaseBrightness;

    if (uTopEnable == 1) {
        vec2 d = vTexCoord - uTopRect.xy;
        if (abs(d.x) <= uTopRect.z && abs(d.y) <= uTopRect.w) {
            vec2 local = d / (2.0 * uTopRect.zw) + 0.5;   // overlay rect 空间 0..1
            vec2 t = local;
            if (uTopMirror == 1) t.x = 1.0 - t.x;
            t = rotQuad(t, uTopRotation);

            vec4 overlayColor = sampleTopFiltered(t);
            float alphaOut = blackKeyAlpha(overlayColor.rgb, overlayColor.a);
            float gain = uOverlayAlpha * uOverlayBrightness;
            vec3 enhanced = clamp(boostSaturation(overlayColor.rgb, uOverlaySaturation) * 1.12, 0.0, 1.0);
            vec3 overlayRGB = enhanced * alphaOut * gain;

            float mask = dimMask(local);
            float baseDim = 1.0 - uOverlayAlpha * mask * uDimStrength;
            color = clamp(color * baseDim + overlayRGB, 0.0, 1.0);
        }
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
