// Native adjustment ground truth for `tests/test_editor_finishing.py`.
//
// Runs Compositor's real C grader kernel (`Rendering/LevelsPixels.c`, pinned at
// 11d8d7a) over a 256-step RGB ramp plus low-alpha edge colours, for several
// Exposure / Levels settings and Invert. The kernel is the exact numeric
// contract: it un-premultiplies each pixel, interpolates the 256-entry curve
// table, and rounds `result * alpha` back into premultiplied bytes. The Python
// compositor grades float straight-alpha, so this fixture pins the divergence.
//
// Build on macOS (needs the pinned C kernel next to it):
//   curl -sL https://github.com/... | tar xz   # or copy from third_party
//   cp .../Compositor-11d8d7a.../Compositor/Rendering/LevelsPixels.{c,h} .
//   printf '#include "LevelsPixels.h"\n' > bridge.h
//   swiftc -import-objc-header bridge.h cg_adjustment_fixture.swift LevelsPixels.c -o cg_adj
//   ./cg_adj > tests/fixtures/cg_adjustment_reference.txt
import Foundation

func premultiplied(_ r: Int, _ g: Int, _ b: Int, _ a: Int) -> [UInt8] {
    func p(_ c: Int) -> UInt8 { UInt8(Int((Double(c) * Double(a) / 255.0).rounded())) }
    return [p(r), p(g), p(b), UInt8(a)]
}

// 256-step ramp (i, 255-i, (i*7)%256) all opaque, then low-alpha edges.
var colours: [(Int, Int, Int, Int)] = []
for i in 0...255 { colours.append((i, 255 - i, (i * 7) % 256, 255)) }
colours += [(200, 100, 50, 1), (255, 0, 0, 128), (0, 255, 0, 64),
            (128, 128, 128, 200), (64, 192, 32, 255), (10, 20, 30, 0)]

struct Range { var black = 0.0; var white = 255.0; var gamma = 1.0; var ob = 0.0; var ow = 255.0 }

func clamp(_ n: Double, _ r: ClosedRange<Double>, _ fallback: Double) -> Double {
    n.isFinite ? min(r.upperBound, max(r.lowerBound, n)) : fallback
}
func levelApply(_ value: Double, _ r0: Range) -> Double {
    var r = r0
    r.black = clamp(r.black, 0...254, 0)
    r.white = clamp(r.white, (r.black + 1)...255, 255)
    r.gamma = clamp(r.gamma, 0.1...9.99, 1)
    r.ob = clamp(r.ob, 0...255, 0)
    r.ow = clamp(r.ow, 0...255, 255)
    let input = min(1, max(0, (value * 255 - r.black) / (r.white - r.black)))
    return (r.ob + pow(input, 1 / r.gamma) * (r.ow - r.ob)) / 255
}
// Native applies ranges[0] (combined RGB) after the per-channel range.
func levelsTable(_ composite: Range, _ channel: Range) -> [Float] {
    (0...255).map { Float(levelApply(levelApply(Double($0) / 255, channel), composite)) }
}
func exposureTable(_ e: Double, _ o: Double, _ g: Double) -> [Float] {
    let scale = pow(2, e)
    return (0...255).map { index in
        let encoded = Double(index) / 255
        var linear = encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4)
        linear = pow(max(0, linear * scale + o), 1 / g)
        let output = linear <= 0.0031308 ? linear * 12.92 : 1.055 * pow(linear, 1 / 2.4) - 0.055
        return Float(min(1, max(0, output)))
    }
}

let ident = Range()
var sweeps: [(String, [Float])] = []
for (name, e, o, g) in [("exp_1", 1.0, 0.0, 1.0), ("exp_neg2_gamma2", -2.0, 0.1, 2.0),
                        ("exp_gamma05", 0.0, -0.2, 0.5)] {
    let t = exposureTable(e, o, g)
    sweeps.append((name, Array(repeating: t, count: 3).flatMap { $0 }))
}
sweeps.append(("lv_rgamma2", (0..<3).flatMap { ch -> [Float] in
    levelsTable(ident, ch == 0 ? Range(gamma: 2) : ident) }))
var red = Range(); red.black = 30; red.white = 220; red.gamma = 2.5; red.ob = 10; red.ow = 240
var comp = Range(); comp.black = 20; comp.white = 200; comp.gamma = 0.5; comp.ob = 5; comp.ow = 255
sweeps.append(("lv_clamp_gamma",
               [levelsTable(comp, red), levelsTable(comp, ident), levelsTable(comp, ident)].flatMap { $0 }))
var red2 = Range(); red2.ob = 20; red2.ow = 235
sweeps.append(("lv_outputrange",
               [levelsTable(ident, red2), levelsTable(ident, ident), levelsTable(ident, ident)].flatMap { $0 }))

for (name, tables) in sweeps {
    var px = colours.map { premultiplied($0.0, $0.1, $0.2, $0.3) }.flatMap { $0 }
    px.withUnsafeMutableBufferPointer { levels_apply($0.baseAddress!, colours.count, tables) }
    print(name + " " + px.map { String(format: "%02x", $0) }.joined())
}
var inv = colours.map { premultiplied($0.0, $0.1, $0.2, $0.3) }.flatMap { $0 }
for i in 0..<colours.count {
    let a = Int(inv[i * 4 + 3])
    for ch in 0..<3 { inv[i * 4 + ch] = UInt8(max(0, a - Int(inv[i * 4 + ch]))) }
}
print("invert " + inv.map { String(format: "%02x", $0) }.joined())
