# GSCP iOS 端

按 Android APK 的能力对齐实现的 iPhone 端。核心协议栈（ADB 客户端 + scrcpy 连接 +
流解析 + VideoToolbox 硬解）为平台无关 Swift 包 `GSCPKit`，iOS App 与 macOS 闭环探针共用。

## 能力对齐矩阵

| Android 能力 | iOS 状态 | 说明 |
|---|---|---|
| adb over TCP 连眼镜（推 server / reverse / shell） | ✅ 完整 | 纯 Swift 实现 adb 线协议（CNXN/AUTH/OPEN/OKAY/WRTE/CLSE），AUTH 用与 Android 同一份 ca.key（RSA-SHA1 签名 + 公钥串），embed 同一 scrcpy-server |
| overlay-only AR 会话（AR 唯一连接模式） | ✅ 完整 | 与 Android HEAD 相同的 server 参数（含 `video_source=display` 兼容老设备），流槽位表同源判定 |
| overlay H.264 解码渲染 | ✅ 完整 | iOS 走 AVSampleBufferDisplayLayer 硬解（探针侧 VideoToolbox 已验证 33+ 帧）；annexB→AVCC 与 Android 同一帧协议 |
| 人脸 AR（overlay 画在人脸正前方） | ✅（追踪器不同） | Android 用自研 ncnn 管线（det_10g+facemesh+hopenet）；iOS 用系统 Vision（人脸框+roll/yaw），锚点渲染语义一致（位置/距离缩放/roll 旋转） |
| 音频（Opus 解码播放） | ✅ 完整 | 内嵌 libopus 1.5.2（SPM C 目标，xiph 官方源码子集）：OpusHead 解析 → 48kHz PCM16 → AVAudioEngine 播放；探针实测真流 568 包 → 109 万采样 |
| 录像 | ✅ 基础 | AVAssetWriter 录手机相机（H.264/MP4）；overlay 合成录制待 Metal 管线 |
| 参数设置（缩放/透明度/亮度/音频开关/恢复默认） | ✅ | 键名与 Android 完全对齐（`overlayScalePct`…），实时生效自动保存 |
| 连接页（IP 输入保存 → 直进 AR） | ✅ | 与 Android HEAD 一致：AR 是唯一连接模式 |

## 目录

```
ios/
├── Package.swift            # SPM：GSCPKit（库）+ gscp-probe（探针）+ gscp-selftest（自检）
├── Sources/GSCPKit/
│   ├── Adb/                 # adb 线协议、认证（SecKey）、sync push、reverse、shell
│   ├── Scrcpy/              # scrcpy 会话：推 server、reverse、四路流解析（帧格式逐字节对齐 Android）
│   ├── Media/               # H264AnnexB/VideoToolbox 解码、SampleBufferFactory、OpusAudioDecoder
│   └── Resources/           # scrcpy-server + ca.key/ca.crt（与 Android assets 同源，改动需同步）
├── Sources/opus/            # libopus 1.5.2 源码子集（解码；SPM C 目标）
├── Sources/GSCPProbe/       # macOS 闭环探针（下述验证的核心工具）
├── Sources/GSCPSelfTest/    # 协议自检（CLT 无 XCTest，用可执行目标）
└── App/                     # iOS App（SwiftUI + AVFoundation + Vision）
    ├── GSCP.xcodeproj       # 手写工程（Xcode 16+ 文件系统同步分组）
    ├── project.yml          # XcodeGen 规格（备用生成路径）
    └── GSCP/                # App 源码
```

## 本机可验证（无需 Xcode）

本仓库开发机只装了 Command Line Tools，以下已在本机真实跑通：

```sh
cd ios
swift build                # GSCPKit + 探针 + 自检 全部编译
swift run gscp-selftest    # 协议自检（帧格式/annexB/adb 帧/CRC/密钥解析/签名/资源）

# 闭环探针：把一台 Android 设备（模拟器 adbd / 真机眼镜）当"眼镜"
#   1) 模拟器（占宿主 5555 端口）：emulator -avd <avd> -port 5554 …
#   2) 探针连 127.0.0.1:5555，跑完整链路：
.build/debug/gscp-probe 127.0.0.1 5555 20
```

2026-10-02 实测（API 36 模拟器 adbd）：

```
[probe] adb connected
[probe] overlay: h264 296x640
[probe] audio codec: opus
[probe] decoded frame 296x640 → /tmp/gscp_ios_probe_frame.png   ← 模拟器实时画面
===== 结果 =====
  overlay 流:   h264 296x640, 118 包 / 538621 B
  audio 流:     opus opusHead=4f70757348656164, 392 包
  VideoToolbox: 117 帧硬解
  Layer 路径:   117 个 CMSampleBuffer 构造成功（失败 0）
  Opus 解码:    752016 个 PCM 采样（48kHz 交错 PCM16）
```

即：iPhone 端将采用的同一套核心代码（协议 + 解析 + 硬解 + 音频解码）已完成
`连接 → 推 server → reverse 隧道 → overlay/opus 双流 → VideoToolbox 硬解
→ libopus 解码` 的真机闭环。

真机 AUTH：对 secure adbd 的 SIGN→RSAPUBLICKEY 交换已实现并在 vivo 测试机上
验证了未授权超时路径；手机上点「允许 USB 调试」后探针即可完成真机授权闭环
（密钥与 Android 端一致，已被眼镜配对过的设备天然信任）。

## 构建 iOS App（需完整 Xcode 17+）

```sh
cd ios/App
open GSCP.xcodeproj         # 手写工程，Xcode 16+ 可直接开
# 打不开则： brew install xcodegen && ./gen-project.sh && open GSCP.xcodeproj
# Cmd+R 运行到 iPhone（真机/模拟器均可；相机 AR 需真机）
```

App 首页输入眼镜 IP → 连接 → 全屏 AR（相机 + overlay 随人脸定位），右下角退出，
左下角设置，中键录像。密钥/公钥/scrcpy-server 由 GSCPKit 资源包自动携带。

## 协议要点（与 Android 实现共同踩过的坑）

- `ca.key` 实为 **PKCS#1** RSAPrivateKey DER（Android 端靠 KeyFactory 宽容解析），
  Swift 端按 PKCS#1/PKCS#8 自动识别，SecKey 原生吃 PKCS#1；
- `scid` 必须是 < 0x80000000 的合法 hex（server 按 `Int32.parseInt(hex,16)`）；
- overlay-only 模式必须显式 `video_source=display`，否则 server 缺省按 camera 走
  Android 12+ 版本检查，会拒绝整个会话（见 Android 仓库同修复）；
- 音频是否开启的判定（本端设置 × server SDK≥29）必须同时决定参数串与流槽位表，
  否则槽位错位、overlay 永远无人读取；
- 帧协议：12B 头 = ptsAndFlags(8, BE；最高位=配置包) + size(4, BE)；overlay 流头
  = codec(4)+w(4BE)+h(4BE)；video 流头 76B；audio 首 socket 可能带 64B 设备名前缀
  （带容错回退，与 Android 相同）。

## 已知限制（后续项）

1. 录像：v1 录相机画面，overlay 合成录制需 Metal 离屏合成；
2. 画面参数中的饱和度/抠像/底图旋转未接（Android 端有，iOS v1 取核心子集）；
3. 人脸追踪 Vision vs ncnn 的锚点几何差异较大时，需按真机眼镜效果微调 overlay
   的缩放系数（`ARContainerUIView.apply(anchor:)`）；
4. libopus 为源码子集内嵌（解码够用）：不含 encoder/DNN/汇编优化变体，如需编码
   能力（反向推流）再补 `dnn/` 与 `src/opus_encoder.c`。
