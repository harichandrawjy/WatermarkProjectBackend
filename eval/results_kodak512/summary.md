# Watermark evaluation

24 photographs + 0 synthetic images x 21 operations = 504 verifications.

Sources: eval/datasets/kodak, every image centre-cropped to 512x512.

Sources are checked before use: anything already carrying a watermark is
refused, and byte-identical duplicates are dropped, so no image can be
double-watermarked or silently counted twice.

## 1. Imperceptibility

| image | type | size | PSNR (dB) | SSIM |
|---|---|---|---|---|
| kodim01 | photo | 512x512 | 41.54 | 0.99864 |
| kodim02 | photo | 512x512 | 41.37 | 0.99304 |
| kodim03 | photo | 512x512 | 40.72 | 0.98199 |
| kodim04 | photo | 512x512 | 41.14 | 0.99196 |
| kodim05 | photo | 512x512 | 41.73 | 0.99882 |
| kodim06 | photo | 512x512 | 40.9 | 0.99185 |
| kodim07 | photo | 512x512 | 41.44 | 0.99402 |
| kodim08 | photo | 512x512 | 41.53 | 0.99836 |
| kodim09 | photo | 512x512 | 40.64 | 0.98341 |
| kodim10 | photo | 512x512 | 40.9 | 0.98828 |
| kodim11 | photo | 512x512 | 41.4 | 0.9941 |
| kodim12 | photo | 512x512 | 41.07 | 0.99177 |
| kodim13 | photo | 512x512 | 41.56 | 0.99874 |
| kodim14 | photo | 512x512 | 41.54 | 0.99759 |
| kodim15 | photo | 512x512 | 41.19 | 0.99175 |
| kodim16 | photo | 512x512 | 40.97 | 0.98912 |
| kodim17 | photo | 512x512 | 41.28 | 0.9934 |
| kodim18 | photo | 512x512 | 41.56 | 0.99614 |
| kodim19 | photo | 512x512 | 41.37 | 0.99259 |
| kodim20 | photo | 512x512 | 39.61 | 0.98208 |
| kodim21 | photo | 512x512 | 40.85 | 0.98489 |
| kodim22 | photo | 512x512 | 41.31 | 0.99205 |
| kodim23 | photo | 512x512 | 40.81 | 0.98597 |
| kodim24 | photo | 512x512 | 41.45 | 0.99693 |
| **mean (photographs)** | | | **41.16** | **0.99198** |
| **mean (all)** | | | **41.16** | **0.99198** |

## 2. Robustness and fragility

`correct` = the verdict matched what the operation should produce.
Photographs are reported separately, since synthetic extremes are there to
probe the limits rather than to represent typical input.

| operation | should be | correct (photos) | correct (all) | mean BER | mean blocks flagged |
|---|---|---|---|---|---|
| none (control) | authentic | 24/24 | 24/24 | 0.0001 | 0.0 |
| PNG re-save | authentic | 24/24 | 24/24 | 0.0001 | 0.0 |
| JPEG q95 | authentic | 18/24 | 18/24 | 0.0013 | 0.3 |
| JPEG q90 | authentic | 18/24 | 18/24 | 0.0018 | 0.3 |
| JPEG q80 | authentic | 17/24 | 17/24 | 0.0023 | 0.4 |
| JPEG q70 | authentic | 14/24 | 14/24 | 0.0031 | 0.5 |
| JPEG q60 | authentic | 13/24 | 13/24 | 0.0038 | 0.5 |
| JPEG q50 | authentic | 8/24 | 8/24 | 0.0061 | 0.9 |
| noise sigma=1 | authentic | 24/24 | 24/24 | 0.0002 | 0.0 |
| noise sigma=3 | authentic | 23/24 | 23/24 | 0.0004 | 0.0 |
| noise sigma=5 | authentic | 23/24 | 23/24 | 0.0007 | 0.1 |
| resize 0.75x | authentic | 10/24 | 10/24 | 0.0001 | 0.0 |
| resize 0.5x | authentic | 0/24 | 0/24 | 0.0004 | 0.0 |
| brightness +1 | authentic | 23/24 | 23/24 | 0.0004 | 0.1 |
| brightness +5 | authentic | 15/24 | 15/24 | 0.0094 | 2.4 |
| contrast x1.01 | authentic | 22/24 | 22/24 | 0.0008 | 0.1 |
| splice / copy-move | tampered | 24/24 | 24/24 | 0.0118 | 3.6 |
| blank region | tampered | 24/24 | 24/24 | 0.0139 | 4.1 |
| single 32x32 edit | tampered | 24/24 | 24/24 | 0.0040 | 1.0 |
| cover 25% of area | tampered | 24/24 | 24/24 | 0.1202 | 42.0 |
| cover 50% of area | tampered | 24/24 | 24/24 | 0.2403 | 82.0 |

## 3. Localisation accuracy

Flagged blocks vs the region actually edited.

| edit | precision | recall | F1 |
|---|---|---|---|
| splice / copy-move | 1.000 | 0.415 | 0.574 |
| blank region | 1.000 | 0.458 | 0.606 |
| single 32x32 edit | 1.000 | 1.000 | 1.000 |
| cover 25% of area | 1.000 | 0.656 | 0.792 |
| cover 50% of area | 1.000 | 0.641 | 0.781 |

## 4. Threshold sweep

Current settings: LSB_BLOCK_TAMPER_RATIO = 0.45, MIN_TAMPER_BLOCKS = 1.

Detection = content-altering edits correctly flagged. 
False alarms = content-preserving operations wrongly flagged.

| ratio | min blocks | detection | false alarms |
|---|---|---|---|
| 0.1 | 1 | 120/120 | 106/384 |
| 0.1 | 2 | 98/120 | 73/384 |
| 0.1 | 3 | 97/120 | 54/384 |
| 0.1 | 4 | 96/120 | 47/384 |
| 0.1 | 6 | 95/120 | 32/384 |
| 0.1 | 8 | 73/120 | 23/384 |
| 0.1 | 12 | 48/120 | 10/384 |
| 0.2 | 1 | 120/120 | 106/384 |
| 0.2 | 2 | 98/120 | 73/384 |
| 0.2 | 3 | 97/120 | 54/384 |
| 0.2 | 4 | 96/120 | 47/384 |
| 0.2 | 6 | 95/120 | 32/384 |
| 0.2 | 8 | 73/120 | 23/384 |
| 0.2 | 12 | 48/120 | 10/384 |
| 0.25 | 1 | 119/120 | 38/384 |
| 0.25 | 2 | 95/120 | 27/384 |
| 0.25 | 3 | 88/120 | 16/384 |
| 0.25 | 4 | 73/120 | 9/384 |
| 0.25 | 6 | 52/120 | 2/384 |
| 0.25 | 8 | 50/120 | 2/384 |
| 0.25 | 12 | 48/120 | 1/384 |
| 0.3 | 1 | 119/120 | 38/384 |
| 0.3 | 2 | 95/120 | 27/384 |
| 0.3 | 3 | 88/120 | 16/384 |
| 0.3 | 4 | 73/120 | 9/384 |
| 0.3 | 6 | 52/120 | 2/384 |
| 0.3 | 8 | 50/120 | 2/384 |
| 0.3 | 12 | 48/120 | 1/384 |
| 0.35 | 1 | 119/120 | 38/384 |
| 0.35 | 2 | 95/120 | 27/384 |
| 0.35 | 3 | 88/120 | 16/384 |
| 0.35 | 4 | 73/120 | 9/384 |
| 0.35 | 6 | 52/120 | 2/384 |
| 0.35 | 8 | 50/120 | 2/384 |
| 0.35 | 12 | 48/120 | 1/384 |
| 0.4 | 1 | 119/120 | 38/384 |
| 0.4 | 2 | 95/120 | 27/384 |
| 0.4 | 3 | 88/120 | 16/384 |
| 0.4 | 4 | 73/120 | 9/384 |
| 0.4 | 6 | 52/120 | 2/384 |
| 0.4 | 8 | 50/120 | 2/384 |
| 0.4 | 12 | 48/120 | 1/384 |
| 0.45 | 1 | 119/120 | 38/384 |  <- current
| 0.45 | 2 | 95/120 | 27/384 |
| 0.45 | 3 | 88/120 | 16/384 |
| 0.45 | 4 | 73/120 | 9/384 |
| 0.45 | 6 | 52/120 | 2/384 |
| 0.45 | 8 | 50/120 | 2/384 |
| 0.45 | 12 | 48/120 | 1/384 |
| 0.5 | 1 | 111/120 | 16/384 |
| 0.5 | 2 | 73/120 | 7/384 |
| 0.5 | 3 | 55/120 | 2/384 |
| 0.5 | 4 | 52/120 | 2/384 |
| 0.5 | 6 | 48/120 | 2/384 |
| 0.5 | 8 | 48/120 | 1/384 |
| 0.5 | 12 | 48/120 | 1/384 |
| 0.55 | 1 | 111/120 | 16/384 |
| 0.55 | 2 | 73/120 | 7/384 |
| 0.55 | 3 | 55/120 | 2/384 |
| 0.55 | 4 | 52/120 | 2/384 |
| 0.55 | 6 | 48/120 | 2/384 |
| 0.55 | 8 | 48/120 | 1/384 |
| 0.55 | 12 | 48/120 | 1/384 |
| 0.6 | 1 | 111/120 | 16/384 |
| 0.6 | 2 | 73/120 | 7/384 |
| 0.6 | 3 | 55/120 | 2/384 |
| 0.6 | 4 | 52/120 | 2/384 |
| 0.6 | 6 | 48/120 | 2/384 |
| 0.6 | 8 | 48/120 | 1/384 |
| 0.6 | 12 | 48/120 | 1/384 |
| 0.65 | 1 | 111/120 | 16/384 |
| 0.65 | 2 | 73/120 | 7/384 |
| 0.65 | 3 | 55/120 | 2/384 |
| 0.65 | 4 | 52/120 | 2/384 |
| 0.65 | 6 | 48/120 | 2/384 |
| 0.65 | 8 | 48/120 | 1/384 |
| 0.65 | 12 | 48/120 | 1/384 |