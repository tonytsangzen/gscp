import SwiftUI

/// 参数设置（能力对齐 Android 设置页子集：overlay 缩放/透明度/亮度、音频开关、
/// 恢复默认；键名与 Android 一致，实时生效自动保存）。
struct SettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Overlay") {
                    sliderRow("缩放", value: settings.overlayScalePct, range: 50...150) {
                        settings.setOverlayScalePct($0)
                    }
                    sliderRow("透明度", value: settings.overlayAlphaPct, range: 20...100) {
                        settings.setOverlayAlphaPct($0)
                    }
                    sliderRow("亮度", value: settings.overlayBrightnessPct, range: 50...150) {
                        settings.setOverlayBrightnessPct($0)
                    }
                }
                Section("音频") {
                    Toggle("接收眼镜音频", isOn: Binding(
                        get: { settings.audioEnabled },
                        set: { settings.setAudioEnabled($0) }))
                    Text("通过内嵌 libopus（1.5.2）解码 48kHz 播放；开关在下次连接生效。")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
                Section {
                    Button("恢复默认", role: .destructive) { settings.resetToDefaults() }
                }
            }
            .navigationTitle("设置")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func sliderRow(_ title: String, value: Int, range: ClosedRange<Int>,
                           onChange: @escaping (Int) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text("\(value)%").foregroundColor(.secondary).monospacedDigit()
            }
            Slider(value: Binding(get: { Double(value) },
                                  set: { onChange(Int($0.rounded())) }),
                   in: Double(range.lowerBound)...Double(range.upperBound))
        }
    }
}
