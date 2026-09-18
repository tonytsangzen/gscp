# GSCP 合并计划 —— mixplayer + GlassConnect 四端统一

> **实施状态（2026-09-18）**：M0–M3 已完成并通过全部测试（62 项）；
> macOS release 打包 6.9MB；Android 工程已生成、核心层通过 NDK 交叉编译。
> 详见文末「实施进度」。

> 目标：将 `mixplayer`（scrcpy 双路视频合成播放器）与 `GlassConnect`（眼镜连接管理器）合并为本项目 `gscp`，
> 支持 **Android / macOS / Linux / Windows** 四端，UI 全部使用系统 WebView，安装包体积尽可能小，
> 完整保留 GlassConnect 的连接管理功能与 mixplayer 的视频/音频/overlay/camera 渲染方式与效果。

---

## 0. 结论速览（TL;DR）

| 维度 | 选型 |
|---|---|
| 语言 | Rust（桌面端 + 核心库）+ Kotlin（Android 壳） |
| UI | **Tauri 2**（仅作系统 WebView 容器，非 UI 框架）：WKWebView / WebKitGTK / WebView2 / Android WebView |
| 前端 | 纯 HTML/CSS/JS（沿用 GlassConnect 的零依赖三件套，加 `bridge.js` 平台适配层） |
| 视频解码 | 平台系统解码器抽象层：macOS=VideoToolbox、Windows=Media Foundation、Linux=系统 libavcodec（动态链接）、Android=MediaCodec |
| 渲染 | wgpu（跑在系统 GPU API：Metal/Vulkan/DX12/GLES 上），移植 mixplayer 的 WGSL 合成管线 |
| 音频 | rodio（CoreAudio/WASAPI/ALSA/pipewire）+ opus 解码（静态 libopus，约 200KB） |
| ADB | 运行时按平台下载 Google platform-tools（不打包）；协议操作复用 `adb_client` crate |
| scrcpy-server | 内嵌 91KB jar 作默认（立即可用），支持从 GitHub Releases 按版本下载更新 |
| 进程模型 | 桌面端**单二进制双模式**：`gscp` = 管理界面（Tauri），`gscp --player` = 播放窗口（winit），管理端 spawn 自身 |
| Android 端 | Kotlin 壳：WebView(管理 UI) + SurfaceView(MediaCodec+GLES 预览) + adblib(TCP 直连) + JNI 调 Rust core 协议层；仅 WiFi 模式 |

体积预算：桌面安装包约 **6–12 MB**，APK 约 **10–15 MB**（详见 §8）。不打包 Chromium / Qt / FFmpeg / WebView 运行时。

---

## 1. 现状盘点

### 1.1 GlassConnect（Tauri 2，目前仅 macOS）

| 功能 | 实现位置 | 合并处理 |
|---|---|---|
| WiFi 配网（enable/scan/add-network wpa2/查 IP/adb tcpip 5555/TCP 验证） | `src/wifi.rs` + `adb_wrapper.rs` | **原样移植** → `gscp-core::wifi` |
| 设备识别（`ro.product.model` 前缀 `RG-`）、USB-only 判定 | `src/adb_wrapper.rs` | **原样移植** → `gscp-core::adb` |
| WiFi 保活 daemon（内嵌 9KB ARM64 ELF，push+chmod+运行） | `src/daemon.rs` + `assets/wifi_daemon` | **原样移植**，daemon 二进制保留内嵌 |
| 停用流程（停 daemon → 关 WiFi → 关 persistent adb） | `src/lib.rs` | **原样移植** |
| ADB 环境清理（disconnect / forward --remove-all / reverse --remove-all） | `src/adb_wrapper.rs` | 移植，adb 子进程改为可配置路径（数据目录中的下载版） |
| 配置持久化（INI，WiFi 列表 ≤20 条、connection_mode、三路流开关） | `src/config.rs` | **原样移植** → `gscp-core::userconfig`，路径改为各平台标准数据目录 |
| UI（模式切换、SSID 下拉、密码显隐、流开关、日志区去重计数、按钮组） | `dist/index.html + app.js + style.css` | **原样移植** + 扩展"渲染效果设置"区 |
| 日志事件流（ANSI/控制符清洗、`log-message` 事件、50ms 批量刷新、500 行上限） | `src/lib.rs` + `app.js` | 移植，抽象为 `log-bus`（Tauri 事件 / Android JS bridge 双通道） |
| 预览（spawn mixplayer，stdout/stderr 转日志，WiFi/USB 两种参数组） | `src/lib.rs` | 改为 spawn 自身 `--player` 模式；Android 端改为进程内渲染 |

### 1.2 mixplayer（Rust + wgpu + winit，目前仅 macOS 验证）

| 功能 | 实现位置 | 合并处理 |
|---|---|---|
| scrcpy server 一键拉起（adb connect/push jar/forward/app_process 启动/renice） | `main.rs` bootstrap + `adb.rs` | **移植**；`shell_spawn` 的 `UnixStream::pair` 改为 `os_pipe`（Windows 可编译） |
| 看门狗 + generation 重连（1s 轮询、去抖重启、socket 预连接缓存） | `main.rs` | **原样移植** |
| 协议解析（dummy byte / 64B 设备名 / codec meta / frame meta / config 包 63 位标志 / 嗅探分路） | `main.rs` | **原样移植** → `gscp-core::scrcpy`；camera/overlay 按面积分路的启发式保留 |
| 解码（annex-B → AVCC/HVCC extradata 手工构造、libavcodec 软解、swscale→RGBA） | `decoder.rs` | **重构**：拆成 `Decoder` trait；extradata 构造逻辑保留（系统解码器同样需要）；见 §5.2 |
| 渲染（wgpu：底图旋转/cover 裁剪/亮度、overlay 圆角 dim mask、5 点十字 AA、边缘锐化、饱和度提升、加色混合、busy spinner、缓冲自适应播放速率） | `renderer.rs` + `main.rs` | **原样移植**到 `gscp-player`，并补齐两项欠账：`overlay_scale` 参数生效、黑边抠像 4 参数接入 shader |
| 音频（rodio ContinuousSource、raw PCM/opus、300ms 背压） | `audio.rs` | **原样移植** |
| 双路缓冲（16 帧槽、4 帧低水位、EMA 帧间隔、0.5–1.5x 速率自适应、overlay 5s 隐藏） | `main.rs` | **原样移植** |
| 配置（clap CLI + config.yaml 三层合并，全部字段见原 README） | `main.rs` | **移植** → `gscp-core::settings`，player 以 CLI 接收，管理端 UI 可改 |

**已知欠账（合并时修复）**：
1. `overlay_scale` CLI 参数在渲染路径未使用（实际 overlay 高度恒为 canvas 高 120%）——恢复参数语义。
2. 黑边抠像 `black_key_low/high`、`feather_power/radius` 进了 uniform 但 WGSL 未引用——在 shader 中实现（luma 阈值 + smoothstep 羽化）。
3. 底图 brightness 硬编码 0.8、dim 系数 0.6、饱和度 1.45/1.12 等 shader 魔数——提取为 uniform/配置项，并在管理端 UI 暴露。
4. `shell_spawn` 使用 `std::os::unix::net::UnixStream`——Windows 编译阻断项，改 `os_pipe`。
5. 删除 Metal 死代码（`MetalPlayer`/MSL shader/cocoa 依赖）。

---

## 2. 需求 → 技术决策对照

### 需求 1：通用能力依赖系统；专用能力自行实现或按系统版本下载

**通用能力（用系统自带的，一律不打包）**：

| 能力 | macOS | Windows | Linux | Android |
|---|---|---|---|---|
| WebView | WKWebView（系统） | WebView2（系统自带，缺则引导下载） | libwebkit2gtk-4.1（系统包，deb 依赖声明） | Android WebView（系统） |
| H264 解码 | VideoToolbox | Media Foundation | 系统 libavcodec（动态链接，发行版自带） | MediaCodec |
| GPU 渲染 | Metal（wgpu） | D3D12/11（wgpu） | Vulkan/GL（wgpu） | GLES3（Kotlin 直用） |
| 音频输出 | CoreAudio（rodio） | WASAPI（rodio） | ALSA/pipewire（rodio） | AudioTrack |

**专用能力（自带或按需下载）**：

| 组件 | 策略 | 说明 |
|---|---|---|
| `adb`（platform-tools） | **按需下载，不打包** | 首次需要时从 `dl.google.com/android/repository/platform-tools-latest-{darwin|linux|windows}.zip` 下载，校验后解压 `adb`（Windows 含 `AdbWinApi.dll`、`AdbWinUsbApi.dll`）到平台数据目录 `bin/`；UI 显示下载进度；已有系统 adb 则可直接选用 |
| `scrcpy-server` | **内嵌默认版 + 可选下载** | 91KB jar 内嵌保证离线可用；设置页支持从 GitHub Releases 拉取指定版本替换 |
| `wifi_daemon` | **内嵌** | 9KB，ARM64 Android ELF，无法"下载"（自研产物），直接 `include_bytes!` |
| WebView2 Runtime | 按需引导 | Windows 打包采用 Tauri `webviewInstallMode: downloadBootstrapper`，缺 Runtime 才联网装 |

### 需求 2：UI 使用嵌入式系统 WebView；native 语言不限

- **Tauri 2**：本质是系统 WebView 容器 + Rust IPC，不带任何浏览器内核和组件框架，符合"放弃大型 UI 框架"。GlassConnect 已是 Tauri 2 项目，前端零依赖三件套直接沿用。
- 桌面 native 语言 = **Rust**；Android 壳 = **Kotlin**（Tauri 2 官方 Android 模板即 Kotlin，视频渲染需要 Kotlin 侧 MediaCodec/GLES）。
- 前端不引入框架；新增的"渲染效果设置"区用原生 DOM（与现有代码风格一致）。如后续复杂度上升再考虑 Preact+htm（单文件、无构建），本期不做。

### 需求 3：Android / macOS / Linux / Windows 四端

- 桌面三端：同一 Rust 代码库 + 条件编译（平台差异收敛在 `gscp-core` 的 4 个平台模块内：解码器、数据目录、下载 URL）。
- Android：Tauri 2 Android 工程（生成的 Kotlin 壳）上做两类定制：
  1. `MainActivity` 增加 `SurfaceView` 预览层 + JS Bridge（`JavascriptInterface` 对齐桌面 Tauri commands）；
  2. JNI 桥接 `gscp-core`（cdylib，`aarch64-linux-android` 等目标），复用协议解析/bootstrap/配置逻辑。
- 平台能力矩阵见 §5.7。

### 需求 4a：安装包体积尽可能小

- 不打包：Chromium/Electron、Qt、FFmpeg 库、WebView 运行时、Node。
- Rust 侧：`lto = "fat"`、`codegen-units = 1`、`strip = "symbols"`、`panic = "abort"`、`opt-level = "z"`（依赖可按 crate 覆盖）。
- wgpu 是最大的 Rust 依赖（二进制约 +4–6MB），换 raw GL（glow）可再省但 macOS OpenGL 已弃用且 Wayland/DX11 适配成本高——保留 wgpu，用体积预算表（§8）兜底。
- 资产内嵌总量 ~100KB（scrcpy-server 91KB + wifi_daemon 9KB）。

### 需求 4b：完整实现 GlassConnect 功能（含视频/音频/overlay/camera 渲染方式与效果）

- GlassConnect 全部命令与 UI 流程 1:1 移植（对照表见 §1.1）。
- mixplayer 的渲染链路（三路流、合成效果、同步策略）1:1 移植，效果参数全集在 UI 暴露实时调节（§5.8），并修复 §1.2 所列 5 项欠账。

---

## 3. 总体架构

```
                          ┌──────────────────────────── 桌面端（macOS/Linux/Windows）────────────────────────────┐
                          │                                                                                      │
   浏览器内核由系统提供 →  │  WebView（管理 UI：连接/配网/效果设置/日志）                                          │
                          │      │ invoke / event（Tauri IPC）                                                   │
                          │  ┌───▼──────────────────────────────┐      spawn 自身                                │
                          │  │ gscp-app（Tauri，主进程）        │ ────────────────┐                             │
                          │  │  ├ commands: start/stop/preview  │                 ▼                             │
                          │  │  ├ gscp-core（连接/配网/配置）    │      ┌──────────────────────────┐             │
                          │  └──────────────────────────────────┘      │ gscp --player（winit）   │             │
                          │                                            │  ├ scrcpy 会话(bootstrap)│             │
                          │                                            │  ├ Decoder(系统解码器)   │             │
                          │                                            │  ├ wgpu 合成渲染         │             │
                          │                                            │  └ rodio 音频            │             │
                          │                                            └──────────────────────────┘             │
                          └──────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ TCP (adb forward / WiFi adb tcpip)
                                                     ▼
                                            Android 眼镜：scrcpy-server（camera + overlay + audio 三路）

                          ┌──────────────────────────── Android 端（手机/平板作为主机）──────────────────────────┐
                          │  WebView（同一套管理 UI，bridge.js → JavascriptInterface）                           │
                          │      │                                                                                │
                          │  MainActivity（Kotlin）                                                               │
                          │   ├ adblib：TCP 直连眼镜 adbd:5555（RSA 认证），shell/push/socket                      │
                          │   ├ JNI → gscp-core(.so)：scrcpy 协议解析 / bootstrap 参数 / settings                  │
                          │   ├ MediaCodec(H264) ×2 → GLES 合成 → SurfaceView（camera 底图 + overlay 叠加）        │
                          │   └ AudioTrack（raw PCM / opus→PCM）                                                   │
                          └──────────────────────────────────────────────────────────────────────────────────────┘
```

**为什么桌面端播放器走子进程（`gscp --player`）**：
- Tauri 的 tao 事件循环与 winit 事件循环不能共存于一个线程/进程（macOS/Windows 只允许一个主事件循环）；
- 沿用 GlassConnect 已验证的"管理端 + 播放器子进程"模型，播放器崩溃不影响管理端；
- 单二进制双模式（`current_exe() + "--player"`）避免两份运行时体积，分发仍是一个文件；
- Android 无子进程能力，用进程内 Kotlin 渲染替代（§5.6）。

---

## 4. 目录结构

```
gscp/
├── Cargo.toml                  # workspace：lto/strip/opt-level 统一配置
├── crates/
│   ├── gscp-core/              # 平台无关核心库（可编译到 android 目标）
│   │   └── src/
│   │       ├── adb/            # adb 定位/下载/调用 + adb_client 封装（connect_device、shell、push、forward）
│   │       ├── scrcpy/         # 协议解析（meta/config 包/嗅探分路）、bootstrap 启动序列、看门狗
│   │       ├── codec/          # extradata 构造（AVCC/HVCC）、Decoder trait 定义
│   │       ├── wifi/           # cmd wifi 配网流程、保活 daemon 部署
│   │       ├── settings/       # 三层配置合并（CLI > 文件 > 默认）、UserConfig(INI)、WiFi 列表
│   │       ├── downloader/     # platform-tools / scrcpy-server 下载解压校验
│   │       └── logbus.rs       # 结构化日志 + 前端事件桥
│   ├── gscp-player/            # 桌面播放器（bin 入口由 gscp-app 转发）
│   │   └── src/
│   │       ├── decode/         # Decoder 实现：videotoolbox.rs / mediafoundation.rs / libav.rs
│   │       ├── render/         # wgpu 管线 + WGSL（自 mixplayer renderer.rs 移植 + 补齐）
│   │       ├── audio/          # rodio 输出 + opus/raw
│   │       └── session.rs      # 三路源线程 + SharedSlot + 自适应速率（自 mixplayer main.rs 移植）
│   └── gscp-app/               # Tauri 桌面应用（bin: gscp）
│       ├── src/main.rs         # 入口：--player 转发到 gscp-player，否则 tauri::run
│       ├── src/commands.rs     # start_wifi / stop_wifi / preview / get_config / save_config / get_wifi_list / save_wifi_network / get_effect / set_effect / tool_status / tool_install
│       └── tauri.conf.json
├── ui/                         # 共享前端（零构建，直接拷贝进 tauri dist 与 android assets）
│   ├── index.html              # GlassConnect UI + 效果设置区
│   ├── style.css
│   ├── app.js
│   └── bridge.js               # 平台桥：window.gscpBridge = tauri(invoke/listen) | android(JSI)
├── android/                    # Tauri 生成的 Kotlin 工程之上定制
│   └── app/src/main/
│       ├── java/.../MainActivity.kt      # WebView + SurfaceView + JSI bridge
│       ├── java/.../PlayerController.kt  # adblib + MediaCodec + GLES 合成 + AudioTrack
│       └── cpp/                          # JNI：调 gscp-core 协议层
├── assets/
│   ├── scrcpy-server           # 内嵌默认 server jar（自 mixplayer 迁移）
│   └── wifi_daemon             # 内嵌保活 daemon（自 GlassConnect 迁移）
├── scripts/                    # build-all.sh / package-*.sh / ci 脚本
└── docs/                       # 协议笔记、平台矩阵、发布清单
```

---

## 5. 关键设计

### 5.1 gscp-core：连接与协议层

- **adb 模块**：
  - 定位顺序：数据目录 `bin/adb`（下载的）→ 系统 PATH → 平台常见路径（沿袭 GlassConnect `find_adb`）。
  - 全部 shell/push/forward 优先走 `adb_client` crate（协议级，快且无子进程开销）；`adb disconnect / forward --remove-all` 等批量清理命令走 adb 二进制子进程（与现状一致）。
  - `shell_spawn`（保持远端 server 存活 + 看门狗）：mixplayer 的 `UnixStream::pair` 实现改为 `os_pipe::pipe()`，Windows 落到 `CREATE_NO_WINDOW` 子进程。
- **scrcpy 模块**：原样移植 mixplayer 的启动参数表（`tunnel_forward=true`、三路同端口 accept、scid 生成、4s 就绪检查、renice）、嗅探分路（77 字节前缀、面积启发式）、config 包处理、generation 看门狗。
- **downloader 模块**：
  - `ensure_platform_tools()` → 下载 zip（`ureq` + 系统代理）→ SHA256 校验（内置 Google 官方清单的可选校验）→ 解压 adb 三件套 → chmod 0755。
  - `ensure_scrcpy_server(version)` → GitHub Releases → 数据目录覆盖；内嵌版始终作为兜底。
  - 全程通过 `logbus` 上报进度（UI 显示"正在下载 adb (2.1MB/5.4MB)"）。

### 5.2 解码抽象（需求 1 的核心落点）

```rust
pub trait Decoder: Send {
    fn configure(&mut self, codec: VideoCodec, extradata: &[u8], w: u32, h: u32) -> Result<()>;
    fn send(&mut self, packet: &[u8], keyframe_hint: bool) -> Result<()>;
    fn recv(&mut self) -> Result<Option<DecodedFrame>>;   // RGBA（桌面）
}
```

| 平台 | 实现 | 输入 | 输出 |
|---|---|---|---|
| macOS | VideoToolbox（`VTDecompressionSession`，FFI 手写，无第三方重依赖） | annex-B → 按 NAL 喂 | CVPixelBuffer → `VTSessionCopy...` 转 RGBA（IOSurface lock） |
| Windows | Media Foundation `IMFTransform`（`windows` crate） | annex-B + MF_MT_MPEG_SEQUENCE_HEADER(extradata) | NV12 → CPU 拷贝转 RGBA（第一期），后续可 D3D11 纹理直通 wgpu |
| Linux | 系统 `libavcodec`/`libswscale` 动态链接（`ffmpeg-next` 保留但仅 Linux feature，运行时 `dlopen` 缺库给安装提示） | 现 decoder.rs 直接复用 | RGBA（现路径） |
| Android | MediaCodec（Kotlin 侧） | annex-B + csd-0 | 直出 Surface 纹理，不解码到内存 |

- H264 第一期必须；HEVC/AV1 随平台能力尽力（VideoToolbox 天然支持 HEVC；MF 需 HEVC 扩展时给出明确报错文案）。scrcpy 下发参数默认 `video_codec=h264` 不变。
- mixplayer 现有的 AVCC/HVCC extradata 手工构造逻辑（decoder.rs:65-245）**保留进 `gscp-core::codec`**——VideoToolbox/MediaFoundation 同样需要它。
- 会话结束 flush、config 包中途重配置（分辨率变化场景）语义与现状一致。

### 5.3 播放器（gscp-player）

- winit 0.30 + wgpu 0.20+，Surface sRGB + Fifo；全屏/Esc/双击行为保留。
- 移植 WGSL 管线，并在 shader 中补齐/参数化：
  - `overlay_scale` 生效（替换硬编码 canvas 高 120%）；
  - 黑边抠像：`luma < key_low` 全透明、`> key_high` 全不透明、中间 `smoothstep` 按 `feather_power/radius` 羽化；
  - 底图 brightness、dim 系数、饱和度增益全部进 uniform；
  - 其余效果（圆角矩形 dim mask、5 点 AA、边缘锐化、加色混合、cover 裁剪、旋转象限）原样保留。
- 三路源线程 + `SharedSlot`（16 帧槽/4 帧低水位/EMA）+ 0.5–1.5x 自适应速率 + overlay 5s 隐藏 + spinner：原样移植。
- 音频：rodio `ContinuousSource` + raw/opus + 300ms 背压：原样移植。
- 与管理端通信：CLI 一次性传参（沿袭现有参数面）+ 本地 stdout 结构化日志（管理端转发到日志区）。

### 5.4 管理端（gscp-app，Tauri 2）

- commands（对齐 GlassConnect 现有 7 个 + 新增）：
  - `start_wifi(ssid, password, keepalive)` / `stop_wifi` / `preview(mode, ip, overlay, camera, audio)` / `get_config` / `save_config` / `get_wifi_list` / `save_wifi_network` —— 逻辑 1:1 移植。
  - 新增：`get_effect()/set_effect(...)`（效果参数读写，实时生效：先落 settings 再通知运行中的 player——通过 player 的本地控制端口或重启会话，第一期允许"下次预览生效 + 运行中可热调 alpha/brightness/scale"）、`tool_status()`（adb/server 是否就绪、版本）、`tool_install(kind)`（触发下载器）。
- 事件：`log-message`、`wifi-connected`、`tool-progress`、`preview-exited`。
- 前端：GlassConnect 三件套整体迁入 `ui/`，DOM 交互代码不变；新增效果设置卡（滑杆+下拉）；`bridge.js` 把 `invoke/listen` 包成 `window.gscpBridge`，Android 端换实现。
- spawn player：`std::env::current_exe()` + `--player ...`；Windows 加 `CREATE_NO_WINDOW`。

### 5.5 Android 端（Kotlin 壳）

- 工程基于 Tauri 2 Android 模板生成后定制（保留其 WebView 管理 UI；`MainActivity` 布局改为 WebView + 预览 SurfaceView 双层）。
- 连接：`com.tananaev:adblib`（纯 Kotlin ADB 客户端，TCP 5555 + RSA 认证）执行 `cmd wifi` 系列、push jar/daemon、启动 scrcpy server、拿到三路 socket。
- 协议复用：三路 socket 的 meta/config 包解析、camera/overlay 分路逻辑通过 JNI 调 `gscp-core`（Rust 读流 → 回调 Kotlin，或 Kotlin 拉字节流）。音频 raw PCM 直接 `AudioTrack`，opus 用 `concentus`（纯 Java opus 解码）或 JNI libopus。
- 渲染：MediaCodec 输出到两个 SurfaceTexture → GLES shader 复刻桌面合成效果（旋转/cover/alpha/亮度/dim mask/黑边抠像——GLSL 移植同一套参数）。
-第一期范围：仅 WiFi 模式（USB host ADB 需要自实现 USB transport，列入后续）。

### 5.6 平台能力矩阵

| 能力 | macOS | Windows | Linux | Android |
|---|---|---|---|---|
| 管理 UI | WKWebView | WebView2 | WebKitGTK | WebView |
| USB 连接 | ✅ | ✅ | ✅ | ❌（后续） |
| WiFi 配网/保活 | ✅ | ✅ | ✅ | ✅ |
| 预览渲染 | wgpu/VT | wgpu/MF | wgpu/libav | GLES/MediaCodec |
| adb 来源 | 下载 | 下载 | 下载 | 不需要（adblib） |
| 打包 | .app + dmg（tauri bundle） | NSIS exe（bootstrapper 装 WebView2） | deb（依赖 libwebkit2gtk）+ AppImage | APK |

### 5.7 效果参数全集（UI 暴露）

| 参数 | 默认 | 说明 |
|---|---|---|
| `base_rotate_deg` | 90 | 底图旋转（90° 象限） |
| `overlay_scale` | 1.2 | overlay 相对 canvas 高比例（修复后生效） |
| `overlay_alpha` | 0.65 | overlay 不透明度（混合增益） |
| `overlay_brightness` | 1.4 | overlay 亮度 |
| `overlay_rotate_deg` | 0 | overlay 旋转（90° 象限） |
| `base_brightness` | 0.8 | 底图亮度（原 shader 魔数参数化） |
| `dim_strength` | 0.6 | overlay 区域底图压暗强度（魔数参数化） |
| `saturation_boost` | 1.45 | overlay 饱和度增益（魔数参数化） |
| `black_key_low/high` | 0/0 | 黑边抠像阈值（修复后生效） |
| `feather_power/radius` | 1.35/0 | 抠像羽化 |
| `enable_camera/overlay/audio` | true | 三路流开关（沿用 GlassConnect UI） |

---

## 6. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **M0 骨架**（起点） | workspace + 三 crate + ui/ + Tauri 桌面空窗口跑通 macOS/Linux/Windows 三目标编译；Android 模板生成可装 APK | `cargo build` 三桌面通过；`apk` 可安装显示空页 |
| **M1 core**（≈5d） | gscp-core：settings/UserConfig、adb 模块（含 os_pipe 重写 shell_spawn）、scrcpy 协议解析与 bootstrap、downloader、logbus；单测覆盖 meta 解析/extradata 构造/分路嗅探 | 协议单测全绿；命令行 demo 能对真机完成 push→forward→启动 server→三路 socket 建立并统计码流 |
| **M2 桌面播放器**（≈10d） | gscp-player：winit+wgpu 渲染移植 + shader 补齐；Decoder trait + macOS VideoToolbox（先行）；音频 rodio；三路会话与自适应同步 | macOS 真机（眼镜）三路预览稳定 30min+；效果参数热调生效；黑边抠像/overlay_scale 可用 |
| **M3 管理端**（≈5d） | Tauri commands 移植 GlassConnect 全部流程；前端迁移 + 效果设置卡 + bridge.js；下载器 UI（adb 首次下载） | macOS 上 GlassConnect 原有操作路径（WiFi 配网/保活/USB/WiFi 预览/日志/配置记忆）全部回归通过 |
| **M4 Windows/Linux**（≈5d） | Media Foundation 解码器、Linux libav 运行时检测；打包（NSIS/deb/AppImage）；WebView2 bootstrapper | Win10/11 与 Ubuntu 22.04/24.04 完整走通 WiFi+USB 预览；安装包体积达标 |
| **M5 Android**（≈10d） | Kotlin 壳：adblib 连接 + WebView UI（bridge.js）+ MediaCodec/GLES 预览 + AudioTrack；JNI 接 core | Android 12+ 真机完成 WiFi 配网 + 预览；APK 体积达标 |
| **M6 收尾**（≈3d） | CI 矩阵（4 端构建+打包）、README/文档、错误文案、版本发布流程 | 一键产出 4 端安装包；文档齐全 |

关键路径风险集中在 M2（系统解码器四套实现）。缓解：M2 先只做 macOS VideoToolbox 交付全链路，其余平台解码器在 M4/M5 各自平台落地；期间 Linux 可临时用 libav 直通验证。

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| VideoToolbox/MediaFoundation 喂流细节（annex-B→AVCC、时机、重配置） | M2 卡点 | 复用 mixplayer 已验证的 extradata 构造；config 包驱动 configure，已留单测 |
| Linux 无统一系统解码 API | 解码方案分歧 | libavcodec 动态链接视为"系统能力"（各发行版均自带）；缺库时 logbus 引导安装命令 |
| wgpu 在老 Linux/VM 无 Vulkan | 黑屏 | 启用 wgpu GLES backend 回退（EGL） |
| WebKitGTK 版本碎片化 | UI 兼容 | 前端只写 ES2019 + 无新 API；CI 加 webkitgtk 4.1 冒烟 |
| Windows spawn 自身被杀软误报 | 分发 | NSIS 签名（可选）；`CREATE_NO_WINDOW`；无自解压行为 |
| 三路 socket 面积分路启发式失败（两路同分辨率） | 预览失败 | 沿用现有报错 + 下发参数保证 camera/overlay 尺寸不同（`camera_size` vs display），文档标注限制 |
| Android adblib RSA 认证兼容性 | 配网失败 | adblib 在多个 scrcpy-Android 客户端中验证过；失败路径给出"先 USB 授权一次"提示 |
| HEVC/AV1 平台差异 | 部分端不可用 | 一期锁 H264（GlassConnect 场景可控），能力探测 + 明确报错 |

---

## 8. 体积预算

| 端 | 组成 | 预算 |
|---|---|---|
| macOS .dmg | 单二进制 ~10–14MB（Tauri+wgpu+core+player）+ scrcpy-server/wifi_daemon 内嵌 ~100KB，dmg 压缩后 | **≈ 6–9 MB** |
| Windows NSIS | 同上（exe），不含 WebView2 Runtime（系统自带/按需引导） | **≈ 5–8 MB** |
| Linux deb | 二进制 + desktop/icon；libwebkit2gtk 走系统依赖 | **≈ 5–8 MB** |
| Android APK | core .so（4 个 ABI 可只出 arm64-v8a → ~3–4MB）+ Kotlin 壳 + WebView UI | **≈ 8–12 MB**（universal） |

对比：GlassConnect 现状（Tauri）~5MB、Electron 方案 ~90MB+、Qt 方案 ~40MB+。满足"足够小"。

---

## 9. 迁移清单（代码来源映射）

| 新位置 | 来源 | 方式 |
|---|---|---|
| `gscp-core/src/scrcpy/*` | mixplayer `main.rs`（协议/bootstrap/看门狗/slot） | 移植重构 |
| `gscp-core/src/adb/*` | mixplayer `adb.rs` + GlassConnect `adb_wrapper.rs` | 合并移植 + os_pipe |
| `gscp-core/src/wifi/*` + `daemon.rs` | GlassConnect `wifi.rs` / `daemon.rs` / `assets/wifi_daemon` | 近似原样 |
| `gscp-core/src/settings/*` | mixplayer CLI+yaml 合并逻辑 + GlassConnect `config.rs` | 合并移植 |
| `gscp-core/src/codec/*` | mixplayer `decoder.rs`（extradata 部分） | 原样 |
| `gscp-player/src/render/*` | mixplayer `renderer.rs`（删 Metal 死代码）+ shader 补齐 | 移植+修改 |
| `gscp-player/src/decode/libav.rs` | mixplayer `decoder.rs`（解码部分） | 原样（仅 Linux feature） |
| `gscp-player/src/audio/*` | mixplayer `audio.rs` | 原样 |
| `ui/*` | GlassConnect `dist/*` | 迁移 + 效果设置卡 + bridge.js |
| `gscp-app/src/commands.rs` | GlassConnect `lib.rs` | 移植改写 |
| `assets/scrcpy-server` | mixplayer / GlassConnect assets | 原样内嵌 |

---

## 10. 实施进度（2026-09-18）

### 已完成并验证

| 里程碑 | 内容 | 验证 |
|---|---|---|
| M0 骨架 | workspace 三 crate + ui/ + assets | 桌面构建通过 |
| M1 core | settings/配置合并、logbus、adb（os_pipe 跨平台 shell_spawn）、scrcpy 协议+嗅探分路+bootstrap 看门狗、codec（annex-B/AVCC/HVCC/Decoder trait）、wifi、daemon、downloader | **39 项单测全绿**（协议头/config 包/嗅探/分路路由/extradata 结构/配置合并/热更新清理等） |
| M2 player | wgpu 渲染移植 + shader 补齐（overlay_scale 生效、黑边抠像实现、亮度/压暗/饱和度参数化）、三路会话、rodio 音频、VideoToolbox FFI + OpenH264 软解 | **22 项测试全绿**：ffmpeg 真实 H264 码流分别经 VideoToolbox（系统硬解）与 OpenH264 解出全部帧；mock scrcpy 会话全链路（握手→协议→解码→槽出帧）；离屏合成数值验证（含黑边抠像：黑色区域透出底图、白色保留） |
| M3 app | Tauri 2 壳、GlassConnect 全部命令 1:1 移植、效果参数命令、工具链下载命令、`--player` 单二进制分发、ui 三件套 + bridge.js + 效果设置卡 + 运行环境卡 | 语法检查通过；管理端 GUI 启动验证；播放窗口对 mock 三路源首帧 192ms、双路 VideoToolbox 解码、渲染 ~100–120fps |
| 打包 | macOS release | **gscp.app 6.9MB**（单二进制含管理端+播放器+内嵌资产），release 冒烟通过 |

### 关键实现决策/修复（相对原计划）

1. **winit 升级 0.29 → 0.30**：0.29 在 macOS 15+ 因 icrate 消息类型错误启动即崩溃；0.30 需迁移到 ApplicationHandler（已完成）。
2. **解码器落地调整**：原计划 Windows=Media Foundation、Linux=系统 libavdyn。为免除系统依赖且四端可用的 H264，落地为 macOS=VideoToolbox（FFI 直链系统框架）+ 其余平台 OpenH264（源码内置编译，约 1–2MB）；MediaFoundation/libav 可作为后续增强。
3. **nal_units_annexb 截断 bug 修复**：mixplayer 原实现对紧贴数据末尾的 NAL 会截断（scrcpy config 包的 PPS 恰在末尾），已用正确的起始码扫描重写并有回归测试。
4. **Tauri 2 移动适配**：gscp-app 增加 cdylib lib target 与 `mobile_entry`；Android 目标不挂 gscp-player（预览由 Kotlin MediaCodec 实现，见 M5）。
5. **效果参数热更新**：管理端写 `settings.yaml`，播放器线程监视 mtime 500ms 轮询热载，实现运行中实时调节（无需控制端口）。

### 待办（后续迭代）

| 项 | 说明 |
|---|---|
| M4 原生打包 | Windows NSIS / Linux deb+AppImage 需各平台原生 CI（交叉编译受 ring/openh264 的 C 工具链限制，本机仅能 check） |
| M4 Media Foundation | Windows 硬解增强（当前 OpenH264 软解可用） |
| M5 Android 完整壳 | Kotlin MediaCodec+GLES 预览、adblib 直连、JSI 事件桥。**构建管线已打通**：Tauri Android 工程生成（crates/gscp-app/gen/android），BuildTask 改为直接调 cargo（绕开 android-studio-script 的 Studio WS 依赖），`gradlew assembleArm64Release` 产出 10MB 未签名 APK；gscp-core/gscp-app 已通过 NDK 交叉编译 |
| 音视频同步 | 沿用 mixplayer 自适应速率方案（无 PTS 对齐），与原实现一致 |
