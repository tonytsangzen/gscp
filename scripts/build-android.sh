#!/usr/bin/env bash
# Android 一键构建：环境自检 → Rust 预构建 → tauri.settings.gradle → Gradle 打包
#
# 用法：
#   scripts/build-android.sh              构建 arm64 release APK
#   scripts/build-android.sh --install    构建并 adb 安装到已连接设备
#
# 可用环境变量覆盖默认值：ANDROID_HOME、NDK_HOME
set -euo pipefail

cd "$(dirname "$0")/.."

INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

# ── 环境检测 ──────────────────────────────────────────────
if [[ -z "${ANDROID_HOME:-}" ]]; then
  for cand in "$HOME/Library/Android/sdk" "$HOME/Android/Sdk"; do
    if [[ -d "$cand" ]]; then ANDROID_HOME="$cand"; break; fi
  done
fi
[[ -n "${ANDROID_HOME:-}" && -d "$ANDROID_HOME" ]] || { echo "✗ 未找到 Android SDK（设置 ANDROID_HOME）"; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)   PREBUILT=darwin-arm64 ;;
  Darwin-x86_64)  PREBUILT=darwin-x86_64 ;;
  Linux-x86_64)   PREBUILT=linux-x86_64 ;;
  *) echo "✗ 未识别主机 $(uname -s)-$(uname -m)"; exit 1 ;;
esac

if [[ -z "${NDK_HOME:-}" ]]; then
  NDK_HOME="$(ls -d "$ANDROID_HOME"/ndk/* 2>/dev/null | sort -V | tail -n1 || true)"
fi
[[ -n "${NDK_HOME:-}" && -d "$NDK_HOME" ]] || { echo "✗ 未找到 NDK（$ANDROID_HOME/ndk/）"; exit 1; }

TOOLCHAIN="$NDK_HOME/toolchains/llvm/prebuilt/$PREBUILT/bin"
export PATH="$TOOLCHAIN:$PATH"
export CC_aarch64_linux_android=aarch64-linux-android24-clang
export AR_aarch64_linux_android=llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$TOOLCHAIN/aarch64-linux-android24-clang"
echo "✓ SDK:  $ANDROID_HOME"
echo "✓ NDK:  $NDK_HOME"

# ── Rust 库预构建 ─────────────────────────────────────────
cargo build --release --package gscp-app --lib --target aarch64-linux-android

# ── tauri.settings.gradle（gitignored，按 Cargo.lock 生成）──
./scripts/gen-tauri-settings.sh

# ── Gradle 打包（release，正式签名：keys/）────────────────
cd crates/gscp-app/gen/android
./gradlew assembleArm64Release

APK=app/build/outputs/apk/arm64/release/app-arm64-release.apk
[[ -f "$APK" ]] || { echo "✗ APK 未生成"; exit 1; }
echo "✓ APK: $(pwd)/$APK"

if [[ $INSTALL -eq 1 ]]; then
  adb install -r "$APK"
  echo "✓ 已安装到已连接设备"
fi
