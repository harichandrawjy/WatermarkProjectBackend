# Test images for the evaluation

Drop your source photographs in this folder. `robustness_eval.py` reads
everything here, and prints what it accepted and what it rejected.

## Hard requirements

- **Never watermarked.** Do not use anything that has been through Encode, and
  do not use anything downloaded from the app. The harness blind-extracts every
  source and **refuses any file that already contains a watermark** — embedding a
  second watermark over a first would make the results meaningless.
- **At least 512 px on the short side.** Anything smaller does not have enough
  32x32 blocks for localisation to mean much. Larger is fine; the harness
  downscales past 1024 px so a full run stays under a minute.
- **PNG or JPEG.** JPEG is fine and realistic — just be aware the "original" then
  already carries compression artefacts, so a couple of PNG or camera-raw-exported
  sources make the imperceptibility numbers cleaner.
- **Distinct images.** Duplicates are detected by content hash and dropped, since
  the same photo counted twice would silently weight the averages.

## What to include, and why it matters

Watermark behaviour depends strongly on local image statistics, so the set
should span them deliberately rather than being eight photos of the same kind.
Aim for **8-12 images**, roughly one or two of each:

| type | example | why it is in the set |
|---|---|---|
| **Smooth / flat areas** | sky, plain wall, studio backdrop | the hardest case — edits blend in and JPEG smooths the region away |
| **Fine texture** | foliage, gravel, fabric, crowd | watermark hides easily here, but JPEG attacks high frequencies hardest |
| **Hard edges / text** | building, signage, a document | JPEG ringing around edges is a known failure mode |
| **Faces / skin** | portrait | directly relevant to the deepfake use case, and skin is smooth in chroma |
| **Low light / noisy** | night or indoor shot | sensor noise interacts with the chroma-LSB layer |
| **High contrast** | strong sun and shadow | stresses the QIM quantiser near clipping |
| **Low contrast** | fog, overcast, washed out | little signal for the watermark to hide behind |

If you can, include one or two that resemble the use cases the app claims —
an insurance-style damage photo, a document, a CCTV-ish frame. An evaluation
run on the kind of media you say the system is for is far more persuasive than
one run on holiday snaps.

## Naming

Filenames appear verbatim in the results tables, so short descriptive names
read better in a report than `IMG_20260812_143022.jpg`:

```
sky-smooth.jpg      portrait.jpg       document-text.png
foliage-detail.jpg  nightshot.jpg      building-edges.jpg
```
