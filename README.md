# gscp

合并 `mixplayer`（scrcpy 双路视频合成播放器）与 `GlassConnect`（眼镜连接管理器）
的四端应用：**Android / macOS / Linux / Windows**。

连接 RG- 系列安卓眼镜：一键 WiFi 配网/保活、USB/WiFi 双路（camera + overlay）
+ 音频预览，GPU shader 合成渲染，效果参数实时可调。

## 技术架构

- **UI**：系统 WebView（WKWebView / WebKitGTK / WebView2 / Android WebView），
  容器为 Tauri 2；前端是零依赖 HTML/CSS/JS（`ui/`，无框架、无构建步骤）。
- **核心**：`crates/gscp-core` —— ADB 访问、scrcpy 协议与会话守护、WiFi 配网、
  配置持久化、工具链按需下载。同一 crate 供桌面与 Android (JNI) 复用。
- **播放器**：`crates/gscp-player` —— wgpu（Metal/Vulkan/DX12）shader 合成、
  平台系统解码器（macOS=VideoToolbox 硬解；其余平台 OpenH264 软解，H264）、
  rodio 音频（raw PCM / Opus）。
- **进程模型**：单二进制双模式
  - `gscp` → 管理界面（Tauri）
  - `gscp --player …` → 播放窗口（winit），管理端 spawn 自身。

### 依赖策略（需求 1）

| 类型 | 处理 |
|---|---|
| 通用能力 | WebView / H264 解码 / GPU / 音频设备 **依赖系统**，不打包 |
| `adb` | **不打包**；首次需要时按平台下载 Google platform-tools 到应用数据目录 |
| `scrcpy-server` | 91KB jar 内嵌（离线可用），支持按版本从 GitHub 下载 |
| `wifi_daemon` | 9KB 内嵌（自研产物） |
| Opus 解码 | unix 用系统 libopus；Windows 音频回退 raw PCM |

安装包体积：单二进制约 **7 MB**（release，含全部内嵌资产）。

## 界面结构

顶部 **连接 / 设置** 两个 Tab（记忆上次所在页）：

- **连接页**：网络设置（WiFi/USB、SSID、保活）、启用/预览/停用、日志输出
- **设置页**：视频流配置（Overlay/Camera/Audio）、渲染效果、运行环境（adb 状态与下载）

## 渲染效果（预览中实时生效）

底图旋转/亮度/cover 裁剪，overlay 缩放/透明度/亮度/旋转/圆角背景压暗/
5 点十字抗锯齿/边缘锐化/饱和度提升/加色混合，黑边抠像（luma 阈值 +
羽化曲线 + 阈值外扩）。全部参数在设置页"渲染效果"区调节，写入
`settings.yaml`，播放器监视文件热更新。

## 构建与测试

```bash
# 全量测试（62 个：协议/配置单测 + VideoToolbox/OpenH264 真实码流
# + mock scrcpy 会话 + 离屏 shader 数值验证）
./scripts/test-all.sh

# 无眼镜端到端演示（mock 三路源）
./scripts/demo.sh

# macOS 打包（.app / .dmg）
./scripts/package-macos.sh app
```

依赖：Rust 1.85+、Node（仅语法检查）、macOS 需要 Xcode CLT；
Linux 需要 `libwebkit2gtk-4.1-dev`；Windows 需要 MSVC + WebView2。
本机 ffmpeg 用于测试码流生成（可选）。

## CI

`.github/workflows/ci.yml`，推送到 GitHub 即自动执行：

| Job | Runner | 内容 | 产物 |
|---|---|---|---|
| desktop (macOS) | macos-latest | 测试 + 打包 | .dmg + .app.zip |
| desktop (Linux) | ubuntu-latest | 测试 + 打包 | .deb + .AppImage |
| desktop (Windows) | windows-latest | 测试 + 打包 | NSIS 安装器 .exe |
| android | ubuntu-latest | arm64 构建 | 未签名 .apk |

- 桌面矩阵在每个平台先跑 `cargo test --workspace`（安装 ffmpeg 使解码/会话集成测试完整执行），再 `npx @tauri-apps/cli@2 build` 打包。
- Android 复用仓库内 `crates/gscp-app/gen/android`（已纳入版本控制，含定制的 BuildTask.kt——直接调 cargo，绕开 android-studio-script 的 Studio WS 依赖），需 JDK 17 + runner 预置 NDK（r25+，含 API 24 wrapper）。
- 产物在 Actions 页面的 Artifacts 区下载；APK 未签名，发布前需自行签名。

## 目录

```
crates/gscp-core     平台无关核心库（可编译到 android 目标）
crates/gscp-player   播放器（渲染/解码/音频/会话）
crates/gscp-app      Tauri 桌面壳 + --player 分发
ui/                  前端（bridge.js 适配 Tauri / Android JSI）
assets/              内嵌 scrcpy-server、wifi_daemon
scripts/             测试/打包/演示脚本
docs/PLAN.md         合并计划与里程碑
```

## 平台状态

| 平台 | 状态 |
|---|---|
| macOS | ✅ 完整验证（VideoToolbox 硬解、双路合成、WiFi/USB、.app 打包） |
| Windows | ✅ 代码就绪（OpenH264/soft、raw 音频）；需原生 CI 打包验证 |
| Linux | ✅ 代码就绪；需原生环境（webkitgtk）打包验证 |
| Android | ✅ 核心库 NDK 交叉编译通过，**arm64 APK 构建成功**（未签名）；Kotlin 视频预览壳见 PLAN.md M5 |

### Android 构建

```bash
export ANDROID_HOME=~/Library/Android/sdk
export NDK_HOME=$ANDROID_HOME/ndk/<版本>
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export PATH="$ANDROID_HOME/ndk/<版本>/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH"
export CC_aarch64_linux_android=aarch64-linux-android24-clang
export AR_aarch64_linux_android=llvm-ar

cd crates/gscp-app/gen/android
./gradlew assembleArm64Release   # 产出 app-arm64-release-unsigned.apk
```
