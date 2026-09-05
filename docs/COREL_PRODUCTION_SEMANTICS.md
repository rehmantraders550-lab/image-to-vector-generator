# Corel Production Semantics Integration

This document records an additive extension to the validated Poster Vector Rebuilder pipeline. It does **not** replace or loosen the existing reconstruction, Corel compatibility, prepress, or CI defaults.

## Locked baseline preserved

The following established settings remain unchanged:

- editable master: `artwork_master.svg`
- stable top-level layer architecture:
  - `00_BACKGROUND`
  - `10_HERO`
  - `20_BRAND`
  - `30_DECORATION`
  - `40_ICONS`
  - `90_PREPRESS`
- default bleed: 3 mm
- default target resolution: 300 PPI
- process-color press path: CMYK
- ICC-managed press conversion
- PDF target: PDF/X-4-oriented
- editable PDF remains separate from the press PDF
- photographic fallback remains explicitly declared rather than falsely vectorized
- pdfcpu relaxed validation remains the acceptance gate; strict validation remains diagnostic
- current panel detection, hidden-background recovery, semantic reconstruction, OCR, vector fitting, and confidence/truth policies are unchanged

## New optional production semantics

The new layer is metadata-driven. Existing artwork with no special production declarations behaves exactly as before.

### Named spot color

An SVG object may explicitly declare a spot ink:

```xml
data-color-role="spot"
data-spot-name="PANTONE 186 C"
```

A spot name must come from authoritative job/artwork information. CMYK-to-Pantone approximation must not be promoted to an authoritative Pantone specification.

### Special finish

A finish object such as print varnish, UV varnish, relief coating, or printing lacquer must be separate vector geometry and declare:

```xml
data-finish-role="uv-varnish"
data-color-role="spot"
data-spot-name="Printing lacquer"
data-overprint-fill="true"
```

Validation rules:

1. separate vector geometry
2. 100% solid coverage
3. no transparency
4. named spot separation
5. explicit overprint fill

A missing rule becomes a production-semantics blocker for that special-finish job.

## Spot-color safety boundary

The current established press exporter intentionally converts to process CMYK. That route remains unchanged.

Therefore, when named spot colors or special finishes are declared:

- the ordinary CMYK export can still be generated for proofing/intermediate use;
- the pipeline must **not** claim that this CMYK PDF preserves named spot separations;
- `press_ready` remains false until a dedicated separation-aware spot-preserving export and RIP/operator validation path is used.

This prevents a silent regression in the existing CMYK pipeline while also preventing false press-safety claims for spot jobs.

## Bleed interpretation

Objects outside the trim boundary are not automatically errors. Full-bleed artwork is expected to extend through the configured bleed area.

Preflight warnings must therefore be interpreted by production intent:

- intentional bleed extension: allowed
- accidental off-page object: review
- insufficient bleed: blocker when manufacturing dimensions are confirmed

The established 3 mm default remains unchanged.

## Color management and proofing

The existing ICC workflow remains the source of truth for the process-color route.

The production manifest now records:

- CMYK ICC profile
- RGB source profile when present
- target PPI
- configured trim and bleed
- process/spot semantic declarations
- special-finish requirements

For intense colors, unusual papers, unusual substrates, or critical brand colors, gamut/output-condition review is required. Physical/device-specific swatch proofing remains authoritative for critical spot-color matching.

## Text policy

No previous font setting changes.

- editable SVG: retain live/editable text where reconstruction is reliable
- current press route: embed fonts as already implemented
- outlining/curves: explicit delivery option when required by the production workflow
- exact font identity: never guessed when the source does not establish it

## CorelDRAW compatibility

The existing SVG cleanup rules remain intact.

The new policy distinguishes:

1. syntactically valid SVG
2. current automated Corel-friendly profile
3. full CorelDRAW open/import visual and structural regression testing

The third is the strongest compatibility proof and should be added as a future environment-specific acceptance fixture when a runnable CorelDRAW test environment is available.

## New output

Every delivery now includes:

`production_manifest.json`

The manifest records inherited baseline settings plus optional production semantics. It is additive and does not replace:

- `reconstruction_report.json`
- `preflight_report.json`
- `artwork_master.svg`
- `artwork_editable.pdf`
- `artwork_press.pdf`
- `artwork_proof.png`

## Preflight classification rule

Conservative diagnostics remain valuable, but warnings are not automatically blockers.

A finding becomes blocking when it contradicts the declared production intent or an established acceptance requirement. This preserves the existing policy for conservative gates while adding explicit handling for spot/finish jobs.
