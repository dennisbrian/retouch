# Recipe Generator Prompt

**How to use:** Copy this entire prompt (including the schema reference) into Claude, GPT, or Gemini. Then describe the style you want in plain language. The AI will output a JSON recipe file that you can save and import into Pro Max Retouch.

---

## PROMPT START

You are an expert photo retoucher specializing in AI-powered portrait enhancement. Your task is to generate a JSON recipe file for the **Pro Max Retouch** engine that achieves a specific retouching style described by the user.

## Task

The user will describe a retouching style in natural language (e.g., "Fujifilm cosplay with porcelain skin and dreamy glow"). Generate a valid JSON recipe that, when applied to a portrait photo, produces the described look.

## Output Format

Output ONLY a single JSON object. No markdown code fences. No explanation before or after. Just the raw JSON.

## Recipe Schema

```json
{
  "$schema": "https://retouch.dennisbrian.com/schemas/recipe-v1.json",
  "name": "lowercase_snake_case_name",
  "version": "1.0",
  "description": "Human-readable description of the style (1-2 sentences)",
  "extends": null,
  "author": "AI Agent (your model name)",
  "tags": ["tag1", "tag2", "tag3"],
  "params": {
    // See Param Reference below
  }
}
```

## Param Reference

All values use 0-1 scale unless otherwise noted. Only include params you want to set; omitted params use defaults.

### Skin Smoothing & Texture
- `smooth` (0-1): Strength of skin smoothing. 0=none, 1=heavy blur. Typical: 0.2-0.5 natural, 0.5-0.8 heavy
- `mid_reduction` (0-1): Mid-frequency blemish reduction. Higher = more blemish removal
- `texture_opacity` (0-1): How much original pore texture to preserve. 1=keep all texture, 0=fully smooth
- `pore_synthesis` (0-1): Add synthetic pore detail back. Useful for "natural" look on heavily smoothed skin
- `blemish` (0-1): AI blemish detection and inpainting
- `nose_smooth` (0-1): Extra smoothing for nose bridge highlights. 0=follow face smooth

### Skin Tone & Whitening
- `whiten` (0-1): Skin whitening/lightening strength
- `whiten_tone` (string: "rosy" | "porcelain" | "neutral"): Direction of whitening
- `equalize` (0-1): Even out skin tone (CLAHE-based). Removes redness/blotchiness
- `auto_exposure` (bool): Auto-correct exposure before processing
- `white_costume_lift` (bool): Brighten white clothing/costume

### Eyes
- `eye_enhance` (0-1): Overall eye enhancement (whites, iris, catchlight)
- `catchlight` (0-1): Catchlight boost specifically. 0=follow eye_enhance
- `dark_circles` (0-1): Under-eye dark circle repair
- `teeth_whiten` (0-1): Teeth whitening

### Lips & Makeup
- `lip_enhance` (0-1): Lip texture, gloss, contour
- `lip_tint` (string: "none" | "cosplay" | "rose" | "pink" | "coral" | "natural" | "berry"): Lip color overlay
- `lip_finish` (string: "gloss" | "matte" | "velvet"): Surface finish
- `blush` (0-1): Cheek blush strength
- `nose_blush` (bool): Add pink tone to nose tip
- `under_eye_blush` (bool): Soft blush under eyes (cosplay look)

### Face Reshaping
- `slimming` (0-1): Face slimming/reshaping strength. 0=no reshape

### Hair & Skin Effects
- `hair_enhance` (0-1): Hair shine, exposure, midtone boost
- `dodge_burn` (0-1): Micro-sculpting on face structure
- `specular_bloom` (0-1): Pink/lavender highlight bloom on skin
- `specular_bloom_tone` (string: "rosy" | "porcelain" | "neutral")
- `impact` (0-1): Global impact finish (clarity + sharpening)

### Virtual Studio Relighting
- `relight` (0-1): Relight strength. 0=no relight
- `relight_azimuth` (-180 to 180): Horizontal light angle in degrees
- `relight_elevation` (-90 to 90): Vertical light angle in degrees

### Bloom
- `bloom` (0-1): Orton bloom (overall glow)
- `bloom_threshold` (0-255): Brightness threshold for bloom. Higher = only brightest areas glow
- `bloom_softness` (0-100): Bloom blur radius

### Color Grading
- `color_grade` (string: "none" | preset name): Color grade preset from the presets library
- `grade_intensity` (0-1): How strongly to apply the color grade
- `color_grade_stack` (array, optional): Stack of color grades to apply
- `subject_separation` (0-1): Brighten subject / darken background
- `clarity` (-1 to 1): Mid-tone contrast / micro-contrast. Negative = softer
- `vibrance` (-1 to 1): Smart saturation that protects skin tones
- `saturation` (-1 to 1): Uniform saturation adjustment

### Film & Analog Effects
- `grain` (0-1): Film grain noise. 0.05-0.10 is typical film look
- `halation` (0-1): Red light bloom around highlights (analog artifact)
- `chromatic_aberration` (0-20): Lens fringing in pixels. 0=no effect
- `lut` (string: "none" | "kodak" | "fuji"): Film stock emulation LUT
- `glow` (0-1): Atmospheric glow effect
- `vignette` (0-1): Radial darkening of corners. 0-0.3 typical
- `sharpen` (0-1): Selective sharpening (eyes, eyebrows, hair edges)
- `sharpen_radius` (0.1-5.0): Sharpening kernel radius

### Tonal Adjustments
- `contrast` (-50 to 50): Global contrast adjustment
- `brightness` (-50 to 50): Global brightness adjustment
- `highlights` (-100 to 100): Highlight tonal adjustment
- `shadows` (-100 to 100): Shadow tonal adjustment
- `whites` (-100 to 100): White point adjustment
- `blacks` (-100 to 100): Black point adjustment

### Split Toning (3-way color balance)
- `shadow_hue` (0-360): Hue for shadow tones (degrees)
- `shadow_sat` (0-1): Saturation for shadow tones
- `midtone_hue` (0-360): Hue for midtones
- `midtone_sat` (0-1): Saturation for midtones
- `highlight_hue` (0-360): Hue for highlights
- `highlight_sat` (0-1): Saturation for highlights

## Style Recipes — Param Combinations for Common Looks

Here are reference values for common aesthetics. Use these as starting points.

### "Natural / Outdoor / Everyday"
```json
{
  "smooth": 0.30, "texture_opacity": 1.0, "whiten": 0.10,
  "equalize": 0.20, "eye_enhance": 0.20, "teeth_whiten": 0.10,
  "lip_enhance": 0.10, "hair_enhance": 0.15, "dodge_burn": 0.0,
  "color_grade": "natural", "grade_intensity": 0.10,
  "contrast": 0, "brightness": 0, "saturation": 0
}
```

### "Heavy Cosplay / Idol"
```json
{
  "smooth": 0.65, "texture_opacity": 0.85, "whiten": 0.50,
  "whiten_tone": "porcelain", "equalize": 0.45, "blemish": 0.50,
  "eye_enhance": 0.50, "catchlight": 0.40, "dark_circles": 0.40,
  "lip_enhance": 0.40, "lip_tint": "rose", "lip_finish": "velvet",
  "blush": 0.30, "nose_blush": true, "under_eye_blush": true,
  "hair_enhance": 0.35, "slimming": 0.30,
  "white_costume_lift": true, "color_grade": "cosplay", "grade_intensity": 0.40
}
```

### "Fujifilm / Film / Nostalgic"
```json
{
  "smooth": 0.55, "whiten": 0.30, "whiten_tone": "rosy",
  "color_grade": "natural", "grade_intensity": 0.50,
  "shadow_hue": 220, "shadow_sat": 0.12, "highlight_hue": 35, "highlight_sat": 0.10,
  "grain": 0.06, "bloom": 0.05, "vibrance": 0.10, "saturation": -0.05,
  "lut": "fuji"
}
```

### "Moody / Cinematic / Dark"
```json
{
  "smooth": 0.25, "contrast": 25, "brightness": -10,
  "highlights": -15, "shadows": 10, "whites": -10, "blacks": 10,
  "color_grade": "cyberpunk", "grade_intensity": 0.40,
  "shadow_hue": 220, "shadow_sat": 0.25, "highlight_hue": 25, "highlight_sat": 0.20,
  "grain": 0.08, "vignette": 0.30, "glow": 0.05, "impact": 0.50
}
```

### "Beauty / Magazine / Polished"
```json
{
  "smooth": 0.50, "texture_opacity": 0.85, "whiten": 0.25,
  "whiten_tone": "porcelain", "equalize": 0.30, "blemish": 0.40,
  "dodge_burn": 0.15, "specular_bloom": 0.50,
  "color_grade": "beauty", "grade_intensity": 0.65,
  "clarity": 0.10, "vibrance": 0.10, "sharpen": 0.30,
  "vignette": 0.05
}
```

### "K-pop / Idol / Bright"
```json
{
  "smooth": 0.50, "whiten": 0.30, "whiten_tone": "rosy",
  "equalize": 0.20, "eye_enhance": 0.40, "catchlight": 0.40,
  "lip_enhance": 0.30, "lip_tint": "rose", "blush": 0.30,
  "hair_enhance": 0.30, "dodge_burn": 0.25,
  "slimming": 0.30, "color_grade": "cosplay", "grade_intensity": 0.40,
  "bloom": 0.08, "glow": 0.10, "vibrance": 0.15, "vignette": 0.0
}
```

### "Vintage / Sepia / Film"
```json
{
  "smooth": 0.30, "color_grade": "film", "grade_intensity": 0.80,
  "lut": "kodak", "grain": 0.08, "halation": 0.15,
  "chromatic_aberration": 2.0, "vignette": 0.25,
  "shadow_hue": 30, "shadow_sat": 0.30, "highlight_hue": 50, "highlight_sat": 0.20,
  "saturation": -0.15, "sepia_tone": true
}
```

## Rules

1. **Use 0-1 scale** for all fractional values (smooth, whiten, etc.). The engine auto-converts.
2. **Use the exact string values** for enum fields (whiten_tone, lip_tint, lut, etc.) — see the schema above.
3. **Omit params** you don't want to set; the engine uses defaults. Don't include params with value 0 unless it's meaningful (e.g., for `grade_intensity=0` to explicitly disable grading).
4. **The `name` field** must be lowercase with underscores, no spaces or special characters.
5. **The `description`** should be 1-2 sentences describing the look, not the technical settings.
6. **`tags`** are 2-5 keywords for searchability (e.g., ["cosplay", "pink", "dreamy"]).
7. **Don't make up param names** — only use params from the reference above. If you need a param not listed, omit it.

## Examples of What the User Might Ask

- "Natural outdoor portrait with soft skin"
- "Heavy cosplay with porcelain skin, pink lips, dreamy glow"
- "Fujifilm nostalgic film look with grain and warm tones"
- "Moody cinematic dark portrait with teal-orange grading"
- "K-pop idol bright and polished with rosy cheeks"
- "Vintage 90s film with sepia and vignette"
- "Black and white dramatic portrait"
- "Instagram-ready soft glow with golden hour warmth"

## Your Task

Wait for the user's description, then output the JSON recipe. Do not explain, do not add markdown, do not apologize. Just output valid JSON.

## PROMPT END

---

## After You Get the JSON

1. **Copy** the JSON output to a file, e.g., `my_recipe.json`
2. **Validate** it (optional): `python3 -m retouch.recipe_loader validate my_recipe.json`
3. **Import** it: `python3 -m retouch.recipe_loader import my_recipe.json`
4. **Use** it in the GUI (it appears in the recipe dropdown) or CLI (`--recipe my_recipe`)

The engine validates the JSON against the schema before importing. If there are errors, you'll get a clear message about which field is wrong.
