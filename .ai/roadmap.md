# Roadmap

## Performance

- Metal/MPS execution provider for MediaPipe/TFLite (GPUDelegate instead of CPU)
- WebGPU Gradio preview — run 800px pipeline client-side, eliminate server round-trip
- Dynamic lazy-load ONNX models in worker pool (only on multi-face dispatch)

## Pipeline

- Neural blemish removal: replace OpenCV fast-marching with lightweight U-Net ONNX
- Multi-subject independent styles: different recipe/style per face in same frame
- 3D relighting gizmo: spherical picker in Gradio → azimuth/elevation mapping

## Architecture

- Decouple `gui.py`: separate layout from callbacks
- In-memory style cache: avoid file I/O for active style profiles
