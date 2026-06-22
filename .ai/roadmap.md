# Engineering Roadmap: Future Enhancements & Optimization

This roadmap outlines planned features, structural refactoring, and performance goals.

## 1. Performance & Hardware Acceleration
- **Metal / MPS execution provider**: Compile custom MediaPipe and TFLite libraries to support native macOS Apple Silicon GPU acceleration (`GPUDelegate`), moving away from the default CPU execution delegate.
- **WebGPU Gradio preview path**: Compile the 800px preview pipeline to WebAssembly/WebGPU, running the interactive slider adjustments completely clientside inside the browser to eliminate server network round-trips.
- **Worker pool lazy-loading**: Optimize `FaceProcessorPool` spin-up latency by implementing dynamic lazy loading of the underlying ONNX models on workers only when multi-face tasks are actively dispatched.

## 2. Retouch Engine Pipeline Features
- **Neural Blemish Removal**: Replace the fast-marching OpenCV inpainting algorithm with a lightweight U-Net ONNX model to execute artifact-free blemish correction.
- **Multi-Subject Independent Styles**: Enhance Stage 2 per-face loops to allow applying different recipes or style transfer references to different subjects within the same frame.
- **Interactive Relighting Gizmo**: Develop a 3D spherical light-direction picker in the Gradio dashboard, mapping horizontal/vertical click coordinates directly to `azimuth` and `elevation` values.

## 3. Architecture & Code Cleanliness
- **Decouple GUI state**: Refactor [gui.py](file:///Applications/htdocs/retouch/gui.py) to extract layout definitions from interactive callbacks, separating view logic from engine invocation.
- **ActiveQuery optimizations**: Restructure style caching to avoid file I/O operations by keeping active styles loaded in an in-memory dictionary.
