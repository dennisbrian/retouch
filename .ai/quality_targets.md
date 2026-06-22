# Quality Targets

## Skin

- No waxy look: bilateral float32 dominant; gaussian blend = `smooth_strength * 0.25`; `mid_reduction` targets isolated blemishes; `pore_synthesis` adds texture at high smooth
- Whitening (`whiten`) restricted to LAB L > 80 pixels only (prevents bruised/purple shadows)
- CLAHE highlight-protection threshold prevents blown catchlights

## Detection & Landmarks

- Symmetrical ROI padding (keep face centered, include hair/ears)
- >95% detection target; RetinaFace fail → fallback full-image MediaPipe pass
- Neck mask blur radius = IED-proportional for seamless chest transition

## Lighting & Specular

- 3D relighting decays to zero at profile ratio >1.7 (no impossible shadows)
- Specular contribution decays from L=220 to L=250
- Lip matte: preserves texture; gloss: lifts catchlights up to 25%
