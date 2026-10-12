import CoreGraphics
import Foundation

let sourceW = 7, sourceH = 5
let outW = 13, outH = 11
let maskBytes: [UInt8] = [
    0, 32, 64, 128, 192, 224, 255,
    255, 224, 192, 128, 64, 32, 0,
    0, 0, 255, 255, 255, 0, 0,
    255, 255, 0, 0, 0, 255, 255,
    0, 64, 128, 255, 128, 64, 0,
]
let provider = CGDataProvider(data: Data(maskBytes) as CFData)!
let mask = CGImage(width: sourceW, height: sourceH, bitsPerComponent: 8,
                   bitsPerPixel: 8, bytesPerRow: sourceW,
                   space: CGColorSpaceCreateDeviceGray(),
                   bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                   decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
let space = CGColorSpace(name: CGColorSpace.sRGB)!
var data = [UInt8](repeating: 0, count: outW * outH * 4)
data.withUnsafeMutableBytes { raw in
    let context = CGContext(data: raw.baseAddress, width: outW, height: outH,
                            bitsPerComponent: 8, bytesPerRow: outW * 4,
                            space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
    context.clear(CGRect(x: 0, y: 0, width: outW, height: outH))
    context.translateBy(x: 0, y: CGFloat(outH))
    context.scaleBy(x: 1, y: -1)
    let center = CGPoint(x: 6.5, y: 5.5)
    let width: CGFloat = 7, height: CGFloat = 5
    context.saveGState()
    context.interpolationQuality = .high
    context.setShouldAntialias(true)
    context.translateBy(x: center.x, y: center.y)
    context.rotate(by: 33 * .pi / 180)
    context.scaleBy(x: 1, y: -1)
    let bounds = CGRect(x: -width/2, y: -height/2, width: width, height: height)
    context.clip(to: bounds, mask: mask)
    context.setFillColor(CGColor(colorSpace: space, components: [1, 1, 1, 1])!)
    context.fill(bounds)
    context.restoreGState()
}
for y in 0..<outH {
    print((0..<outW).map { String(data[(y*outW+$0)*4+3]) }.joined(separator: ","))
}
