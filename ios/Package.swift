// swift-tools-version:5.9
// GSCP iOS 端核心包：ADB 客户端 + scrcpy 连接 + 流解析 + VideoToolbox 解码。
// GSCPKit 平台无关（iOS/macOS 共用）；gscp-probe 是 macOS 闭环探针（对 Android
// 模拟器 adbd 跑 adb→push→reverse→overlay 流→硬解全链路，见 ios/README.md）。
import PackageDescription

let package = Package(
    name: "GSCPKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "GSCPKit", targets: ["GSCPKit"]),
        .executable(name: "gscp-probe", targets: ["GSCPProbe"]),
        .executable(name: "gscp-selftest", targets: ["GSCPSelfTest"]),
    ],
    targets: [
        // libopus 1.5.2（xiph 官方源码子集：celt + silk/float + src，无 DNN/汇编变体）
        .target(
            name: "opus",
            path: "Sources/opus",
            publicHeadersPath: "include",
            cSettings: [
                .define("OPUS_BUILD"),
                .define("PACKAGE_VERSION", to: "\"1.5.2\""),
                .define("VAR_ARRAYS"),
                .headerSearchPath("include"),
                .headerSearchPath("celt"),
                .headerSearchPath("silk"),
                .headerSearchPath("silk/float"),
                .headerSearchPath("src"),
            ]
        ),
        .target(
            name: "GSCPKit",
            dependencies: ["opus"],
            resources: [
                .copy("Resources/scrcpy-server"),
                .copy("Resources/ca.key"),
                .copy("Resources/ca.crt"),
            ]
        ),
        .executableTarget(
            name: "GSCPProbe",
            dependencies: ["GSCPKit"]
        ),
        // 协议自检（CLT 环境无 XCTest/swift-testing 模块，用可执行目标代替 swift test）
        .executableTarget(
            name: "GSCPSelfTest",
            dependencies: ["GSCPKit"]
        ),
    ]
)
