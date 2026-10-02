import Foundation
import Combine

/// 用户设置：键名与 Android 端 ArActivity/MainActivity 完全对齐
/// （同 key 即可在两端保持一致的语义；值仅存各自设备的 UserDefaults）。
@MainActor
final class AppSettings: ObservableObject {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        _ip = defaults.string(forKey: "ip") ?? ""
        _overlayScalePct = defaults.object(forKey: "overlayScalePct") as? Int ?? 100
        _overlayAlphaPct = defaults.object(forKey: "overlayAlphaPct") as? Int ?? 100
        _overlayBrightnessPct = defaults.object(forKey: "overlayBrightnessPct") as? Int ?? 100
        _audioEnabled = defaults.object(forKey: "audioEnabled") as? Bool ?? true
    }

    @Published private(set) var ip: String
    @Published private(set) var overlayScalePct: Int
    @Published private(set) var overlayAlphaPct: Int
    @Published private(set) var overlayBrightnessPct: Int
    @Published private(set) var audioEnabled: Bool

    func saveIp(_ value: String) {
        ip = value
        defaults.set(value, forKey: "ip")
    }

    func setOverlayScalePct(_ v: Int) { overlayScalePct = v; defaults.set(v, forKey: "overlayScalePct") }
    func setOverlayAlphaPct(_ v: Int) { overlayAlphaPct = v; defaults.set(v, forKey: "overlayAlphaPct") }
    func setOverlayBrightnessPct(_ v: Int) { overlayBrightnessPct = v; defaults.set(v, forKey: "overlayBrightnessPct") }
    func setAudioEnabled(_ v: Bool) { audioEnabled = v; defaults.set(v, forKey: "audioEnabled") }

    /// 恢复默认（对齐 Android「恢复默认」：只复位画面参数，保留 IP）。
    func resetToDefaults() {
        setOverlayScalePct(100)
        setOverlayAlphaPct(100)
        setOverlayBrightnessPct(100)
        setAudioEnabled(true)
    }
}
