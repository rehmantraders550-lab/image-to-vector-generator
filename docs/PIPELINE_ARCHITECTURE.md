# Poster Vector Rebuilder — Production Pipeline Architecture v1

## Objective

Turn a single photographed/raster reference artwork into two production outputs:

1. **Editable SVG master** — every recoverable design element separated into named groups and editable primitives, optimized for CorelDRAW/Inkscape/Illustrator interoperability.
2. **Vector press PDF** — print-ready PDF/X-oriented output with CMYK/spot-colour validation, page boxes, separations, and automated QA.

The system is not a blind image tracer. It uses measured source pixels wherever they exist, AI only where information is occluded, and deterministic vector reconstruction for the final artwork.

---

## Governing Accuracy Rule

```text
VISIBLE SOURCE PIXELS  = authority / measurement data
HIDDEN SOURCE PIXELS   = AI inference only
FINAL VECTOR           = deterministic editable reconstruction
PRINT OUTPUT            = validated separately from editable master
```

AI-generated pixels must never overwrite reliable visible source pixels merely to make the image look prettier.

---

# End-to-End Pipeline

## Stage 0 — Intake

**Input:** JPG / PNG / TIFF / camera photograph

Capture:
- source dimensions
- EXIF orientation
- user-supplied final physical size if known
- intended print process if known
- intended substrate if known
- requested bleed if known

Generate immutable `source_original` plus working copies.

---

## Stage 1 — Geometric Normalization

**Primary:** OpenCV

Tasks:
- orientation correction
- poster/page boundary detection
- perspective rectification / homography
- crop to actual artwork
- optional lens-distortion correction
- preserve original pixel coordinates through transform metadata

Outputs:
- `normalized_reference.png`
- `geometry.json`

Do not upscale before source analysis.

---

## Stage 2 — Element Segmentation

**Primary:** SAM 2
**Secondary:** BiRefNet

Create masks for:
- background
- product / hero object
- logos
- text blocks
- badges
- icons
- splashes / organic decoration
- geometric panels
- shadows/highlights where separable

Masks are stored separately and remain editable.

Outputs:
- `masks/*.png`
- `segmentation.json`

---

## Stage 3 — Background Recovery

### 3A. Known-pixel map

OpenCV classifies unoccluded background pixels as authoritative samples.

### 3B. Missing-area inference

**Workflow engine:** ComfyUI

Preferred inpainting options:
1. LaMa — first choice for smooth gradients, walls, repetitive backgrounds and low-semantic fills.
2. BrushNet — optional difficult-mask reconstruction.
3. PowerPaint — optional semantic/object-aware reconstruction.

The inpainted result is a hypothesis, not ground truth.

Outputs:
- `background_known_mask.png`
- `background_inferred.png`
- `background_confidence.json`

---

## Stage 4 — Design Primitive Classification

Each segmented element is routed to the correct reconstruction method.

| Element type | Reconstruction method |
|---|---|
| Smooth colour field | Custom gradient fitter + OpenCV |
| Linear/radial gradient | Native SVG gradient fitting |
| Translucent geometric plane | Polygon/path reconstruction |
| Hard-edged logo/icon | VTracer then path cleanup |
| Monochrome silhouette | Potrace / Inkscape tracing |
| Organic decorative shape | VTracer + path simplification |
| Text | OCR + font analysis + editable text |
| Photographic object | keep as raster only if genuine vector recovery is impossible; flag explicitly |

The pipeline must never pretend a raster photograph is a true vector object.

---

## Stage 5 — Text and Typography Reconstruction

**Primary OCR:** PaddleOCR
**Fallback OCR:** Tesseract
**Font tooling:** fontTools + FontForge

Tasks:
- detect text regions
- recover text content
- estimate baseline, tracking, size, alignment and rotation
- identify likely installed font if available
- reproduce as SVG `<text>` when reliable
- optionally convert a duplicate delivery version to paths

Outputs:
- `text_layers.json`
- editable SVG text objects

Unknown fonts are flagged rather than silently substituted.

---

## Stage 6 — Vector Reconstruction

**Core:** custom SVG builder in this repository
**Optimization:** diffvg
**Tracing:** VTracer

Construct the artwork using a restricted vector primitive set:
- paths / Bézier curves
- rectangles
- ellipses
- polygons
- linear gradients
- radial gradients
- fills/strokes
- opacity
- clipping paths
- editable text
- named groups

Avoid unsupported or fragile SVG effects unless they survive CorelDRAW compatibility testing.

---

## Stage 7 — Perceptual Fit / Error Optimization

Render the generated vector and compare it against authoritative reference regions.

Metrics:
- RGB residual
- CIE Lab / Delta-E where colour-managed measurement is available
- edge alignment error
- structural overlap / mask IoU

Adjust only the relevant parameters:
- gradient vector
- stop locations
- stop colours
- opacity
- polygon coordinates
- Bézier control points
- object transforms

Stop when error converges or improvement becomes visually insignificant.

Never optimize hidden/inpainted areas as though they were known source truth.

---

## Stage 8 — SVG Assembly and CorelDRAW Compatibility

**Primary compatibility validator:** Inkscape CLI + project-specific SVG profile

Required layer/group model:

```text
ARTWORK
├── 00_BACKGROUND
│   ├── Base Gradient
│   ├── Panel 01
│   ├── Panel 02
│   └── Texture
├── 10_HERO
├── 20_BRAND
│   ├── Logo
│   ├── Headline
│   └── Supporting Text
├── 30_DECORATION
├── 40_ICONS
└── 90_PREPRESS
```

Compatibility rules:
- no embedded HTML
- no JavaScript
- no external CSS dependency
- flatten transforms where doing so improves interoperability
- use explicit SVG viewBox
- use simple gradients instead of filter-heavy effects
- preserve logical group names
- keep source raster objects clearly labelled if unavoidable

Primary editable output:
- `artwork_master.svg`

---

## Stage 9 — Colour and Prepress Preparation

**Tools:** Scribus + LittleCMS

Tasks:
- map intended process colours
- identify RGB objects
- apply selected ICC workflow
- preserve or create spot colours where explicitly defined
- establish page size / trim / bleed
- create print-delivery PDF

Important: SVG remains the editable master. The press PDF is a separate controlled export.

Output:
- `artwork_press.pdf`

---

## Stage 10 — Automated PDF / Separation Validation

**Tools:** Ghostscript + qpdf + pdfcpu

Tests:
- PDF structural integrity
- page-box sanity
- TrimBox / BleedBox / MediaBox
- font status
- colour-space inventory
- unexpected RGB
- process separations
- spot/DeviceN separations where present
- raster proof render
- final output dimensions

Generate:
- `preflight_report.json`
- `preflight_report.html`
- `proof_composite.png`
- optional separation proofs

No automatic PASS result should claim press safety for trapping/overprint edge cases that require operator or RIP review.

---

# User Experience Target

```text
UPLOAD REFERENCE IMAGE
        ↓
automatic analysis
        ↓
segmentation + recovery
        ↓
vector reconstruction
        ↓
vector/reference optimization
        ↓
CorelDRAW compatibility cleanup
        ↓
prepress conversion + QA
        ↓
DOWNLOAD
  • artwork_master.svg
  • artwork_press.pdf
  • preview.png
  • preflight_report.html
  • reconstruction_report.json
```

---

# Confidence Model

Every reconstructed element receives a confidence class:

- **A — Measured vector:** directly derived from visible source geometry/colour.
- **B — Strong reconstruction:** mostly measured with limited interpolation.
- **C — Inferred:** substantial source area hidden; AI/interpolation used.
- **D — Approximation:** source information insufficient for exact reconstruction.

The report must expose these classes instead of presenting all reconstructed content as equally exact.

---

# Implementation Order

## Phase 1 — Deterministic foundation
- OpenCV normalization
- source/working-coordinate tracking
- mask format
- SVG compatibility profile
- vector render comparison
- reporting schema

## Phase 2 — Background engine
- background sample extraction
- gradient fitting
- panel detection
- diffvg optimization
- reference error heatmap

## Phase 3 — Segmentation + inpainting
- SAM 2 integration
- ComfyUI API integration
- LaMa first-pass inpainting
- BrushNet/PowerPaint optional routes

## Phase 4 — Foreground vectorization
- VTracer routing
- OCR
- text reconstruction
- path cleanup

## Phase 5 — CorelDRAW delivery
- SVG normalization tests
- editable layer naming
- compatibility regression fixtures

## Phase 6 — Prepress
- Scribus/LittleCMS colour stage
- PDF export
- Ghostscript/qpdf/pdfcpu preflight
- separation proof generation

## Phase 7 — One-click orchestrator
- single job manifest
- queue/retry logic
- final ZIP/delivery directory
- reproducibility metadata


---

# Additive Corel Production-Semantics Contract

The validated reconstruction and prepress baseline above remains authoritative. The Corel production-semantics extension is additive and must not alter existing reconstruction settings or ordinary CMYK jobs.

New delivery metadata:
- `production_manifest.json`

Optional SVG declarations:
- `data-color-role="spot"`
- `data-spot-name="<authoritative spot name>"`
- `data-finish-role="<declared coating/finish>"`
- `data-overprint-fill="true"`

When a special finish is declared, the preflight contract requires separate vector geometry, solid 100% coverage, no transparency, a named spot separation, and explicit overprint intent.

The existing press exporter remains CMYK/PDF/X-4-oriented. It must not claim spot-separation preservation. A declared spot/finish job therefore requires a future separation-aware export/RIP validation path before `press_ready` can be true.

Bleed warnings are classified by intent: geometry extending outside trim for configured bleed is not automatically an error.

See `docs/COREL_PRODUCTION_SEMANTICS.md` for the full compatibility and safety boundary.
