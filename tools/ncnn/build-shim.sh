#!/usr/bin/env bash
# 编译 AR native 管线（ar_face_live.cpp，移植自 ncnn-benchmark face_live.cpp）：
#   tools/ncnn/build-shim.sh android → jniLibs/arm64-v8a/libncnnshim.so
# 依赖：build-android-arm64/ 已构建 libncnn（CMakeCache: Release + NCNN_VULKAN=ON）。
# shim 必须以 -DNCNN_VULKAN=1 编译（ncnn::Net 布局与库一致），并链 vulkan。
set -euo pipefail
cd "$(dirname "$0")"

NCNN_SRC="$PWD/src/src"          # net.h / gpu.h 等头文件
ENGINE="$PWD/ar_face_live.cpp"

ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
NDK_HOME="${NDK_HOME:-$(ls -d "$ANDROID_HOME"/ndk/* 2>/dev/null | sort -V | tail -n1)}"
CXX="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang++"
[[ -x "$CXX" ]] || CXX="$NDK_HOME/toolchains/llvm/prebuilt/darwin-arm64/bin/aarch64-linux-android24-clang++"

# ncnn 静态库带 OpenMP：链接运行时（蓝本 CMake 同款）
OMP_FLAG=""
SYSROOT_DIR="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot"
[[ -d "$SYSROOT_DIR" ]] || SYSROOT_DIR="$NDK_HOME/toolchains/llvm/prebuilt/darwin-arm64/sysroot"
for OMP_LIB in "$SYSROOT_DIR/usr/lib/aarch64-linux-android/libomp.a"; do
  [[ -f "$OMP_LIB" ]] && OMP_FLAG="$OMP_LIB"
done

"$CXX" -shared -fPIC -O2 -std=c++17 \
  -DNCNN_VULKAN=1 -DNCNN_OPENMP=1 -DNCNN_STRING=1 \
  -I"$PWD/include" -I"$NCNN_SRC" -I"$PWD/build-android-arm64/src" "$ENGINE" \
  -L"$PWD/build-android-arm64/src" -lncnn \
  ${OMP_FLAG:+"$OMP_FLAG"} -llog -ldl -landroid -lvulkan \
  -o "$PWD/../../crates/gscp-app/gen/android/app/src/main/jniLibs/arm64-v8a/libncnnshim.so"
echo "✓ jniLibs/arm64-v8a/libncnnshim.so"
