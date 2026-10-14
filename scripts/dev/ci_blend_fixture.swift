// Core Image blend-mode singular-corner fixture (RGBA8, null color space).
//
// Renders CIColorDodge/Burn/VividLightBlendMode over solid (Cb, Cs) probe
// pairs and prints one hex byte per probe, so the Python port's _blend
// corner handling stays pinned to real Core Image behavior. Build with:
//   swiftc -O scripts/dev/ci_blend_fixture.swift -o /tmp/ci_blend_fixture
//   /tmp/ci_blend_fixture > tests/fixtures/ci_blend_reference.txt

import CoreImage
import Foundation

let probes: [(Double, Double)] = [
    (0.0, 1.0), (1.0 / 255.0, 1.0), (0.3, 0.15), (0.3, 0.6),
    (1.0, 0.0), (0.8, 0.25), (0.9, 0.4),
    (0.3, 0.7), (0.95, 0.05), (0.85, 0.15), (0.9, 0.1),
]

let filters = [
    ("color_dodge", "CIColorDodgeBlendMode"),
    ("color_burn", "CIColorBurnBlendMode"),
    ("vivid_light", "CIVividLightBlendMode"),
]

let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])

func solid(_ value: Double) -> CIImage {
    let color = CIColor(red: CGFloat(value), green: CGFloat(value), blue: CGFloat(value), alpha: 1)
    return CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
}

var out = ""
for (name, filterName) in filters {
    guard let filter = CIFilter(name: filterName) else { fatalError("missing \(filterName)") }
    var bytes: [UInt8] = []
    for (cb, cs) in probes {
        filter.setValue(solid(cb), forKey: kCIInputBackgroundImageKey)
        filter.setValue(solid(cs), forKey: kCIInputImageKey)
        guard let image = filter.outputImage else { fatalError("no output") }
        var pixel = [UInt8](repeating: 0, count: 4)
        context.render(image, toBitmap: &pixel, rowBytes: 4,
                       bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBA8, colorSpace: nil)
        bytes.append(pixel[0])
    }
    out += name + " " + bytes.map { String(format: "%02x", $0) }.joined(separator: " ") + "\n"
}
print(out, terminator: "")
