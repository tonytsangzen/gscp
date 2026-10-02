import SwiftUI

@main
struct GSCPApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var session = GlassesSession()

    var body: some Scene {
        WindowGroup {
            ConnectView()
                .environmentObject(settings)
                .environmentObject(session)
                .preferredColorScheme(.dark)
        }
    }
}

/// 主连接页（能力对齐 Android MainActivity：AR 是唯一连接模式——
/// 输入眼镜 IP → 连接 → 直接进入 AR 会话）。
struct ConnectView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var session: GlassesSession
    @State private var ipText = ""
    @State private var showAR = false
    @FocusState private var ipFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                Image(systemName: "gearshape")
                    .foregroundColor(.white.opacity(0.5))
                    .padding(16)
            }

            Spacer()

            VStack(spacing: 12) {
                Text("GSCP 眼镜投屏")
                    .font(.largeTitle.bold())
                Text("请先用桌面端完成眼镜 Wi-Fi 配网，\n并确保手机与眼镜处于同一网络")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)

            Spacer()

            VStack(alignment: .leading, spacing: 20) {
                Text("眼镜 IP 地址")
                    .font(.title3.weight(.medium))
                    .foregroundColor(.secondary)
                TextField("", text: $ipText)
                    .keyboardType(.decimalPad)
                    .textFieldStyle(.plain)
                    .font(.title2)
                    .focused($ipFocused)
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: 10).fill(.white.opacity(0.06)))
                Button {
                    let ip = ipText.trimmingCharacters(in: .whitespaces)
                    guard !ip.isEmpty else {
                        ipFocused = true
                        return
                    }
                    settings.saveIp(ip)
                    session.connect(ip: ip, settings: settings)
                    showAR = true
                } label: {
                    Text("连接")
                        .font(.title3.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(14)
                        .background(RoundedRectangle(cornerRadius: 10).fill(.purple))
                        .foregroundColor(.white)
                }
                statusLine
            }
            .padding(24)

            Spacer(minLength: 60)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear { ipText = settings.ip }
        .fullScreenCover(isPresented: $showAR) {
            ARScreen()
                .environmentObject(settings)
                .environmentObject(session)
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch session.state {
        case .idle:
            EmptyView()
        case .connecting:
            HStack(spacing: 8) {
                ProgressView()
                Text("连接眼镜中…").foregroundColor(.secondary)
            }
        case .streaming(let codec, let w, let h):
            Text("已连接：\(codec) \(w)×\(h)，帧 \(session.overlayFrames)")
                .font(.footnote).foregroundColor(.green)
        case .error(let msg):
            Text(msg).font(.footnote).foregroundColor(.red)
        }
        if !session.audioNote.isEmpty {
            Text("音频：\(session.audioNote)")
                .font(.caption2).foregroundColor(.secondary)
        }
    }
}
