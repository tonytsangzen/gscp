#version 100
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vTexCoord;
varying vec2 vTransformedTexCoordBottom;
varying vec2 vTransformedTexCoordTop;
uniform int uTopEnable;
uniform samplerExternalOES uBottomTexture;
uniform samplerExternalOES uTopTexture;

void main() {
    vec4 bottom = texture2D(uBottomTexture, vTransformedTexCoordBottom);

    if (uTopEnable == 1) {
        // 检查纹理坐标是否在有效范围内
        if (vTransformedTexCoordTop.x >= 0.0 && vTransformedTexCoordTop.x <= 1.0 &&
        vTransformedTexCoordTop.y >= 0.0 && vTransformedTexCoordTop.y <= 1.0) {
            vec4 top = texture2D(uTopTexture, vTransformedTexCoordTop);
            gl_FragColor = bottom + top;
            return;
        }
    }

    gl_FragColor = bottom;
}