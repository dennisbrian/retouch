// Native byte-domain ground truth for kinds 14/15 (Black & White, Color
// Balance): drives Rendering/AdjustPixels.c `adjust_black_white` and
// `adjust_color_balance` at 11d8d7a over a fixed probe set, printing one
// hex pixel per probe per setting. Mirrors cg_float_adjustment_fixture.swift.
//
// Build (needs the pinned C kernel next to it):
//   cp .../Compositor-11d8d7a.../Compositor/Rendering/{AdjustPixels,LensPixels}.{c,h} .
//   printf '#include "AdjustPixels.h"\n' > bridge.h
//   swiftc -import-objc-header bridge.h cg_bw_cb_fixture.swift \
//       AdjustPixels.c LensPixels.c -o cg_bw_cb
//   ./cg_bw_cb > tests/fixtures/cg_bw_cb_reference.txt
import Foundation

// 256-step ramps plus labelled colors, all opaque: byte-exact cases where the
// primary/secondary split and tonal weights change fastest.
var probes: [(Int, Int, Int)] = []
for i in 0...255 { probes.append((i, 255 - i, (i * 7) % 256)) }
probes += [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
           (0, 255, 255), (255, 255, 255), (0, 0, 0), (128, 128, 128), (64, 64, 64),
           (200, 100, 50), (64, 192, 32), (10, 20, 30), (240, 180, 160),
           (30, 90, 200), (120, 30, 180), (180, 200, 40), (90, 40, 40)]

func hex(_ r: Int, _ g: Int, _ b: Int) -> String {
    String(format: "%02x%02x%02xff", r, g, b)
}

func run(_ label: String, _ body: (UnsafeMutablePointer<UInt8>, Int, Int) -> Void) {
    var px = [UInt8]()
    for (r, g, b) in probes { px += [UInt8(r), UInt8(g), UInt8(b), 255] }
    px.withUnsafeMutableBufferPointer {
        body($0.baseAddress!, probes.count, 1)
    }
    for (i, (r, g, b)) in probes.enumerated() {
        let o = (Int(px[i * 4]), Int(px[i * 4 + 1]), Int(px[i * 4 + 2]))
        print("\(label) \(hex(r, g, b)) \(hex(o.0, o.1, o.2))")
    }
}

// --- Black & White -----------------------------------------------------------
// Photoshop defaults (40/60/40/60/20/80, tint off), plus one all-100 mix and a
// tinted case (sepia: hue 40, sat 20). Weights order: r,y,g,c,b,m / 100.
for (label, weights, tint, hue, sat) in [
    ("bw_default", [40.0, 60.0, 40.0, 60.0, 20.0, 80.0], 0, 0.0, 0.0),
    ("bw_unweighted", [100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 0, 0.0, 0.0),
    ("bw_boosted", [200.0, 300.0, -200.0, -100.0, 40.0, 0.0], 0, 0.0, 0.0),
    ("bw_tinted", [40.0, 60.0, 40.0, 60.0, 20.0, 80.0], 1, 40.0, 20.0),
] {
    let w = weights.map { Float($0 / 100) }
    let hs = [Float(hue)], ss = [Float(sat / 100)]
    run(label) { p, width, height in
        w.withUnsafeBufferPointer { wbuf in
            hs.withUnsafeBufferPointer { h in
            ss.withUnsafeBufferPointer { s in
                adjust_black_white(p, width, height, width * 4, wbuf.baseAddress!,
                                   Int32(tint), Double(h[0]), Double(s[0]))
            }}
        }
    }
}

// --- Color Balance -----------------------------------------------------------
// Photoshop-style single-band pushes and a combined case, with and without
// preserve-luminosity. Arrays are cyan/red, magenta/green, yellow/blue in -1..1.
for (label, shadows, midtones, highlights, keep) in [
    ("cb_shadows", [0.5, -0.25, 0.75], [0, 0, 0], [0, 0, 0], 1),
    ("cb_mids", [0, 0, 0], [-0.4, 0.6, -0.2], [0, 0, 0], 1),
    ("cb_highlights", [0, 0, 0], [0, 0, 0], [0.3, 0.8, -0.6], 1),
    ("cb_all_nolum", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0),
    ("cb_all_keep", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1),
] {
    let shadowsF = shadows.map { Float($0) }
    let midtonesF = midtones.map { Float($0) }
    let highlightsF = highlights.map { Float($0) }
    run(label) { p, width, height in
        shadowsF.withUnsafeBufferPointer { s in
        midtonesF.withUnsafeBufferPointer { m in
        highlightsF.withUnsafeBufferPointer { h in
            adjust_color_balance(p, width, height, width * 4,
                                 s.baseAddress!, m.baseAddress!, h.baseAddress!, Int32(keep))
        }}}
    }
}
