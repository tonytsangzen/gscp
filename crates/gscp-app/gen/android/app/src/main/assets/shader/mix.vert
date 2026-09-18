#version 100
precision mediump float;
attribute vec4 aPosition;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;
uniform mat4 uBottomViewMatrix;
uniform mat4 uTopViewMatrix;
uniform int uBottomMirror;
uniform int uTopMirror;
varying vec2 vTransformedTexCoordBottom;
varying vec2 vTransformedTexCoordTop;

void main() {
    gl_Position = aPosition;
    vTexCoord = aTexCoord;

    vec4 bottomCoord = vec4(aTexCoord, 0.0, 1.0);
    bottomCoord = uBottomViewMatrix * bottomCoord;
    if (uBottomMirror == 1) {
        bottomCoord.x = 1.0 - bottomCoord.x;
    }
    vTransformedTexCoordBottom = bottomCoord.xy;

    vec4 topCoord = vec4(aTexCoord, 0.0, 1.0);
    topCoord = uTopViewMatrix * topCoord;
    if (uTopMirror == 1) {
        topCoord.x = 1.0 - topCoord.x;
    }
    vTransformedTexCoordTop = topCoord.xy;
}