# 标准发布流程

## Android APK

**唯一发布入口：`scripts/build-android.sh [--install]`。禁止用裸 `./gradlew assemble…` 或 IDE 出包发布。**

```sh
adb connect <设备IP>:5555           # WiFi 设备先连接（--install 的前提）
scripts/build-android.sh --install  # 构建 arm64 release APK 并安装到已连接设备
scripts/build-android.sh            # 只构建不安装
```

产物：`crates/gscp-app/gen/android/app/build/outputs/apk/arm64/release/app-arm64-release.apk`。

## 为什么必须走脚本（踩坑记录）

| 陷阱 | 后果 | 脚本的对策 |
|---|---|---|
| `libncnnshim.so` **不在 Gradle/Rust 构建链内**（源码 `tools/ncnn/ar_face_live.cpp`，AR 人脸跟踪管线） | 改了 `tools/ncnn/*.cpp` 后绕过脚本构建，APK 静默打包旧库；或手动跑 `build-shim.sh` 与正式出包脱节 → "native 改动没生效/行为不一致"的误判，排查数小时 | 每次构建统一重编 shim |
| Rust 交叉编译环境（CC/AR/linker）缺失 | 裸 `./gradlew` 在 `ring` 等依赖上直接失败（找不到 `aarch64-linux-android24-clang`） | 自动探测 SDK/NDK 路径并注入环境变量 |
| `tauri.settings.gradle` 为 gitignored 生成物 | 依赖变更后缺失或过期 | 按 `Cargo.lock` 统一刷新 |

## 发布前检查清单

- [ ] 通过 `scripts/build-android.sh` 构建（未绕过）
- [ ] 改过 `tools/ncnn/*.cpp`：构建日志里确认 shim 重新编译（`✓ jniLibs/arm64-v8a/libncnnshim.so`）
- [ ] 排查 native 行为问题时先**卸载重装**（防设备上残留旧库造成误判）
- [ ] 真机回归：AR 前摄（跟踪 / overlay 对齐 / 发光 / 录像）、AR 后摄（画面 / 音频）、第一页播放
