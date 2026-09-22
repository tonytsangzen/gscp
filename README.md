# GSCP — 眼镜投屏伴侣

在电脑或手机上观看 AR 眼镜画面的工具：给眼镜配好 Wi-Fi，之后随时随地无线投屏——
眼镜看到的（camera）+ 眼镜显示的（overlay）+ 声音，合成在一个窗口里实时观看。

支持 **Windows / macOS / Linux / Android**。

---

## 下载

到 [Releases](https://github.com/tonytsangzen/gscp/releases/latest) 页面，按平台下载：

| 平台 | 文件 |
|---|---|
| Windows | `gscp_<版本>_x64-setup.exe` |
| macOS（Apple Silicon） | `gscp_<版本>_aarch64.dmg` 或 `.app.zip` |
| Linux（Debian/Ubuntu） | `gscp_<版本>_amd64.deb` |
| Linux（通用） | `gscp_<版本>_amd64.AppImage` |
| Android（arm64） | `gscp_<版本>_aarch64.apk` |

## 桌面端使用（Windows / macOS / Linux）

### 第一次使用：给眼镜配网

眼镜要连上你家 Wi-Fi 才能无线投屏，配网只需做一次：

1. 安装并打开 GSCP。
2. 用 USB 线把眼镜连到电脑。
3. 连接页保持「WiFi」模式，填入 **SSID**（Wi-Fi 名称）和 **密码**
   （也可点 SSID 右侧 ▼ 选择之前保存过的网络）。
4. 点 **「启用」**——程序会自动把 Wi-Fi 信息发给眼镜，眼镜联网后自动获取 IP。
5. 点 **「预览」**——打开播放窗口开始观看。

> 勾选「Wi-Fi 保活」可防止眼镜闲置后自动断开，建议保持开启。

### 日常使用：无线投屏

眼镜开机后会自动连上配好的 Wi-Fi（与电脑同一网络）：

1. 打开 GSCP，确认 IP 地址已自动填入（或手动输入眼镜 IP）。
2. 点 **「启用」**，再点 **「预览」**。

### 有线模式

不想走 Wi-Fi 时，选「有线」模式，USB 直连眼镜后点「启用」→「预览」，
延迟最低，适合调试。

### 结束观看

- 点 **「停用」** 断开连接；直接关窗也可。
- macOS 可在 `~/Library/Application Support/gscp/`、
  Windows 在 `%APPDATA%\gscp\` 找到配置与日志。

## 播放窗口与画面调节

播放窗口同时显示两路画面：**底图**（眼镜 camera 实景）与 **overlay**
（眼镜 UI 图层，叠加在实景上）。

在「设置」页可调整（预览播放中实时生效，自动保存）：

- 视频流配置：overlay / camera / 音频各自开关；
- 渲染效果：叠加透明度 / 亮度 / 缩放 / 旋转、底图亮度 / 旋转、
  背景压暗、饱和度、黑边抠像（下限 / 上限 / 羽化）；
- 调乱了就点 **「恢复默认」**。

## Android 端使用

手机端只负责「连接 + 播放」，**配网请先用桌面端完成**。

1. 手机安装 `gscp_<版本>_aarch64.apk`（需允许安装未知来源应用）。
2. 确认手机与眼镜连接的是**同一个 Wi-Fi**。
3. 打开 GSCP，输入眼镜 IP（就是桌面端配网成功后显示的那个），点「连接」。
4. 播放中：点连接页右上角 **⚙** 调整参数（overlay 缩放 / 透明度 /
   亮度 / 饱和度 / 抠像 / 底图旋转等）；播放为全屏沉浸模式，
   从屏幕边缘滑动可临时呼出系统栏，按返回键断开回到连接页。

## 常见问题

**提示找不到 adb / 下载 platform-tools 失败？**
GSCP 不自带 adb，首次使用会自动从 Google 服务器下载（约 10 MB）。
网络不可访问 `dl.google.com` 时会失败，可在「设置」页点
「下载 platform-tools」重试，或自行安装 adb 后重启 GSCP。

**画面发灰、overlay 像蒙了一层半透明？**
点设置页「恢复默认」。旧版本默认参数带了压暗和半透明叠加，
新版本默认不改变画面，但已保存过的旧配置需要手动恢复一次。

**延迟高？**
优先使用有线模式；无线模式确认眼镜与电脑连的是同一个路由器、
信号良好。若持续异常，把应用数据目录下的 `gscp.log` 发给开发者。

**Windows 双击安装包提示 SmartScreen？**
点「仍要运行」即可——安装包未做代码签名。

**macOS 提示「"gscp"已损坏，无法打开」？**
应用未经 Apple 公证，macOS 会拦截带下载隔离属性的应用。终端执行一条命令
（对已拖入「应用程序」的 GSCP）即可解除：

```sh
xattr -cr /Applications/gscp.app
```

之后再打开就不会再提示。

**Android 提示"连接失败"？**
确认桌面端配网已成功、手机与眼镜在同一 Wi-Fi、IP 输入正确；
眼镜省电断开 Wi-Fi 时重新点亮眼镜再试。

## 从源码编译 Android 版本

前置（一次性）：

- Rust + Android 目标：`rustup target add aarch64-linux-android`
- JDK 17、Android SDK（platform-tools / android-36 / build-tools）、Android NDK（r27+）

一键构建（自动检测 SDK/NDK 路径与主机平台，Rust 预构建 → 生成配置 → Gradle 打包）：

```sh
scripts/build-android.sh              # 构建 arm64 release APK（正式签名）
scripts/build-android.sh --install    # 构建并 adb 安装到已连接设备
```

产物：`crates/gscp-app/gen/android/app/build/outputs/apk/arm64/release/app-arm64-release.apk`（可直接安装）。

手动构建（等价步骤，便于排查）：

```sh
# 1. 环境变量（按本机 NDK 版本/路径调整）
export ANDROID_HOME=$HOME/Library/Android/sdk
export NDK_HOME=$ANDROID_HOME/ndk/28.2.13676358
export PATH=$NDK_HOME/toolchains/llvm/prebuilt/darwin-arm64/bin:$PATH
export CC_aarch64_linux_android=aarch64-linux-android24-clang
export AR_aarch64_linux_android=llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK_HOME/toolchains/llvm/prebuilt/darwin-arm64/bin/aarch64-linux-android24-clang

# 2. Rust 库预构建
cargo build --release --package gscp-app --lib --target aarch64-linux-android

# 3. 生成 tauri.settings.gradle（gitignored，首次/依赖变更后需要）
./scripts/gen-tauri-settings.sh

# 4. Gradle 打包
cd crates/gscp-app/gen/android
./gradlew assembleArm64Release
```

日常增量构建：Rust 代码变更后重跑 2 + 4；纯 Kotlin/资源变更只跑 4。
版本号在三处：`Cargo.toml`、`crates/gscp-app/tauri.conf.json`、
`crates/gscp-app/gen/android/app/tauri.properties`（versionName/versionCode）。

## License

本项目代码以 [MIT License](LICENSE) 授权。

其中捆绑/依赖的第三方组件各自遵循其原始许可，主要包括：

| 组件 | 用途 | 许可 |
|---|---|---|
| scrcpy（scrcpy-server） | 眼镜端码流服务 | Apache-2.0 |
| OpenH264（源码编译） | H.264 软件解码 | BSD-2-Clause |
| libopus | 音频解码（Unix 系统库） | BSD-3-Clause |
| libadb（Android 端 AAR） | adb 协议实现 | GPL-3.0 |
| Tauri / wgpu / winit 等 Rust 生态 | 应用框架与渲染 | MIT OR Apache-2.0 |

> 注意：Android 端因捆绑 GPL-3.0 的 libadb 库，该端应用按 GPL-3.0 授权分发；
> 桌面端为 MIT。OpenH264 为源码编译产物（Cisco 官方二进制另有授权条款，本项目未使用）。
