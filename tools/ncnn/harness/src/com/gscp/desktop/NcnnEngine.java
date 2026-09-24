package com.gscp.desktop;

/** host 离线 harness 用的最小 JNI 声明，与设备端 NcnnEngine.kt 外部函数一致。 */
public class NcnnEngine {
    public native long nativeInit(String param, String bin, boolean useGpu, boolean allowFp16);
    public native void nativeRelease(long handle);
    public native void nativeResetTrack();
    public native int nativePose(long det, long lm, byte[] frame, int w, int h, float[] out);
    public native float[] nativeRun(long handle, String inputName, float[] data, String[] names, int[] dimsOut);
}
