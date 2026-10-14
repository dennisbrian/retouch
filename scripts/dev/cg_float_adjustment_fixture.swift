// Native float-path ground truth for tests/test_editor_finishing.py
// (test_float_adjustment_matches_native_fixture).
//
// Drives two native implementations, pinned at Compositor 11d8d7a:
//   * Rendering/AdjustPixels.c `adjust_camera_raw` — Camera Raw's Light/Color
//     kernel. Temperature/Tint gains come from Document/CameraRaw.swift
//     (`temperatureGain 0.35`, `tintRedBlue 0.15`, `tintGreen 0.30`), applied
//     in LINEAR light with no luminance renormalisation. Contrast pivots at
//     0.5 in encoded sRGB. Vibrance is luminance-anchored with skin-tone
//     protection (not an HSL saturation scale).
//   * Document/HueSaturation.swift `HueSaturationFilter.adjust` — Photoshop's
//     Hue/Saturation. Master-range hue rotation and the Photoshop saturation
//     rule (below 0 multiplies toward gray, above 0 divides by what remains).
//     Reimplemented below verbatim (pure functions, no SwiftUI deps).
//
// Brightness has NO native additive counterpart; the port's additive shift is
// recorded as reference-only (`kind brightness_ref`), and the closest native
// (Hue/Saturation Lightness) is emitted as `brightness_vs_lightness` for the
// deviation report.
//
// Build on macOS (needs the pinned C kernel next to it):
//   curl -sL https://github.com/... | tar xz   # or copy from third_party
//   cp .../Compositor-11d8d7a.../Compositor/Rendering/{AdjustPixels,LensPixels}.{c,h} .
//   printf '#include "AdjustPixels.h"\n' > bridge.h
//   swiftc -import-objc-header bridge.h cg_float_adjustment_fixture.swift \
//       AdjustPixels.c LensPixels.c -o cg_float_adj
//   ./cg_float_adj > tests/fixtures/cg_float_adjustment_reference.txt
import Foundation

// --- Probes: 256-step ramp + labeled colours, all opaque -------------------
var probes: [(Int, Int, Int)] = []
for i in 0...255 { probes.append((i, 255 - i, (i * 7) % 256)) }
probes += [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
           (0, 255, 255), (255, 255, 255), (0, 0, 0), (128, 128, 128), (64, 64, 64),
           (200, 100, 50), (64, 192, 32), (10, 20, 30), (240, 180, 160)]

func hex(_ r: Int, _ g: Int, _ b: Int) -> String {
    String(format: "%02x%02x%02xff", r, g, b)
}

// --- Camera Raw kernel (adjust_camera_raw) ----------------------------------
func cameraRaw(_ label: String, _ gains: (Double, Double, Double),
               _ kwargs: [(String, Double)] = []) {
    var px = [UInt8]()
    for (r, g, b) in probes { px += [UInt8(r), UInt8(g), UInt8(b), 255] }
    var contrast = 0.0, vibrance = 0.0, saturation = 0.0
    for (k, v) in kwargs {
        switch k { case "contrast": contrast = v; case "vibrance": vibrance = v
                   case "saturation": saturation = v; default: break }
    }
    px.withUnsafeMutableBufferPointer {
        adjust_camera_raw($0.baseAddress!, probes.count, 1, probes.count * 4,
                          gains.0, gains.1, gains.2, 0, contrast, 0, 0, 0, 0,
                          vibrance, saturation, 0)
    }
    let params = (kwargs.map { "\($0.0)=\($0.1)" }).joined(separator: ",")
    for (i, (r, g, b)) in probes.enumerated() {
        let o = (px[i * 4], px[i * 4 + 1], px[i * 4 + 2])
        print("\(label) \(params) \(hex(r, g, b)) \(hex(Int(o.0), Int(o.1), Int(o.2)))")
    }
}

// --- Photoshop Hue/Saturation (HueSaturationFilter.adjust, verbatim) -------
func hueSatToHsl(_ r: Double, _ g: Double, _ b: Double) -> (Double, Double, Double) {
    let high = max(r, g, b), low = min(r, g, b)
    let lightness = (high + low) / 2
    let delta = high - low
    guard delta > 0 else { return (0, 0, lightness) }
    let saturation = delta / (1 - abs(2 * lightness - 1))
    var hue: Double
    if high == r { hue = (g - b) / delta }
    else if high == g { hue = (b - r) / delta + 2 }
    else { hue = (r - g) / delta + 4 }
    hue *= 60
    if hue < 0 { hue += 360 }
    return (hue, min(1, saturation), lightness)
}

func hueSatHueToRgb(_ p: Double, _ q: Double, _ t0: Double) -> Double {
    var t = t0
    if t < 0 { t += 1 }
    if t > 1 { t -= 1 }
    if t < 1.0 / 6 { return p + (q - p) * 6 * t }
    if t < 0.5 { return q }
    if t < 2.0 / 3 { return p + (q - p) * (2.0 / 3 - t) * 6 }
    return p
}

func hueSatToRgb(_ hue: Double, _ saturation: Double, _ lightness: Double)
        -> (Double, Double, Double) {
    guard saturation > 0 else { return (lightness, lightness, lightness) }
    let chroma = (1 - abs(2 * lightness - 1)) * saturation
    let sector = hue / 60
    let second = chroma * (1 - abs(sector.truncatingRemainder(dividingBy: 2) - 1))
    let base = lightness - chroma / 2
    let (red, green, blue): (Double, Double, Double)
    switch Int(sector) {
    case 0: (red, green, blue) = (chroma, second, 0)
    case 1: (red, green, blue) = (second, chroma, 0)
    case 2: (red, green, blue) = (0, chroma, second)
    case 3: (red, green, blue) = (0, second, chroma)
    case 4: (red, green, blue) = (second, 0, chroma)
    default: (red, green, blue) = (chroma, 0, second)
    }
    return (min(1, max(0, red + base)), min(1, max(0, green + base)), min(1, max(0, blue + base)))
}

func photoshopAdjustedSaturation(_ saturation: Double, _ amount0: Double) -> Double {
    let amount = min(1, max(-1, amount0 / 100))
    guard amount > 0 else { return max(0, saturation * (1 + amount)) }
    return amount >= 1 ? (saturation > 0 ? 1 : 0) : min(1, saturation / (1 - amount))
}

// Master range, colorize off: shift/sat/lightness are the raw slider values.
func hueSaturation(_ label: String, hue: Double = 0, saturation: Double = 0, lightness: Double = 0) {
    for (r, g, b) in probes {
        let dr = Double(r) / 255, dg = Double(g) / 255, db = Double(b) / 255
        var (h, s, l) = hueSatToHsl(dr, dg, db)
        var lightnessAmount = lightness / 100
        h = (h + hue).truncatingRemainder(dividingBy: 360)
        if h < 0 { h += 360 }
        s = photoshopAdjustedSaturation(s, saturation)
        let amount = min(1, max(-1, lightnessAmount))
        l = amount >= 0 ? l + (1 - l) * amount : l * (1 + amount)
        let out = hueSatToRgb(h, s, min(1, max(0, l)))
        let fr = Int((min(1, max(0, out.0)) * 255).rounded())
        let fg = Int((min(1, max(0, out.1)) * 255).rounded())
        let fb = Int((min(1, max(0, out.2)) * 255).rounded())
        var params = [String]()
        if hue != 0 { params.append("hue=\(Int(hue))") }
        if saturation != 0 { params.append("saturation=\(Int(saturation))") }
        if lightness != 0 { params.append("lightness=\(Int(lightness))") }
        print("\(label) \(params.joined(separator: ",")) \(hex(r, g, b)) \(hex(fr, fg, fb))")
        _ = lightnessAmount
    }
}

// --- Brightness: port reference (no native additive counterpart) ------------
// Port math: out = clip(in + amount/100, 0, 1), rounded to bytes.
func brightnessRef(_ amount: Int) {
    for (r, g, b) in probes {
        func f(_ c: Int) -> Int {
            Int((min(1, max(0, Double(c) / 255 + Double(amount) / 100)) * 255).rounded())
        }
        print("brightness_ref amount=\(amount) \(hex(r, g, b)) \(hex(f(r), f(g), f(b)))")
    }
}

// --- Sweeps ------------------------------------------------------------------
// Saturation (Photoshop Hue/Saturation master): -50, +50, +100
hueSaturation("saturation_native", saturation: -50)
hueSaturation("saturation_native", saturation: 50)
hueSaturation("saturation_native", saturation: 100)
// Vibrance (Camera Raw kernel): +50, -50, +100
cameraRaw("vibrance_native", (1, 1, 1), [("vibrance", 50)])
cameraRaw("vibrance_native", (1, 1, 1), [("vibrance", -50)])
cameraRaw("vibrance_native", (1, 1, 1), [("vibrance", 100)])
// Temperature (Camera Raw gains, linear light): ±30, +60
for t in [30, -30, 60] {
    let w = Double(t) / 100
    cameraRaw("temperature_native", (1 + 0.35 * w, 1, 1 - 0.35 * w))
}
// Tint: ±30
for t in [30, -30] {
    let m = Double(t) / 100
    cameraRaw("tint_native", (1 + 0.15 * m, 1 - 0.30 * m, 1 + 0.15 * m))
}
// Contrast (Camera Raw, pivot 0.5): ±50
cameraRaw("contrast_native", (1, 1, 1), [("contrast", 50)])
cameraRaw("contrast_native", (1, 1, 1), [("contrast", -50)])
// Hue (Hue/Saturation master): +30, -90, +180
hueSaturation("hue_native", hue: 30)
hueSaturation("hue_native", hue: -90)
hueSaturation("hue_native", hue: 180)
// Brightness port reference + nearest native (Hue/Sat Lightness) for the report
brightnessRef(30)
brightnessRef(-30)
hueSaturation("brightness_vs_lightness", lightness: 30)
hueSaturation("brightness_vs_lightness", lightness: -30)
