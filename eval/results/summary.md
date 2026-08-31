# Watermark evaluation

8 photographs + 2 synthetic images x 19 operations = 190 verifications.

Sources are checked before use: anything already carrying a watermark is
refused, and byte-identical duplicates are dropped, so no image can be
double-watermarked or silently counted twice.

## 1. Imperceptibility

| image | type | size | PSNR (dB) | SSIM |
|---|---|---|---|---|
| 6645f90981339 | photo | 1024x672 | 42.19 | 0.99741 |
| beautiful-sunny-day- | photo | 800x576 | 41.84 | 0.99421 |
| dark-city-pictures-7 | photo | 1024x640 | 42.52 | 0.98393 |
| Gravel_RatTrap-IMG_1 | photo | 1024x544 | 42.15 | 0.99753 |
| indonesian-e-passpor | photo | 1024x704 | 42.09 | 0.99928 |
| Lenna_(test_image) | photo | 512x512 | 41.19 | 0.99173 |
| rear-view-group-youn | photo | 800x512 | 41.17 | 0.98646 |
| white-blank-concrete | photo | 736x736 | 42.0 | 0.99367 |
| synth-gradient | synthetic | 512x512 | 39.29 | 0.93791 |
| synth-texture | synthetic | 512x512 | 41.55 | 0.99947 |
| **mean (photographs)** | | | **41.89** | **0.99303** |
| **mean (all)** | | | **41.60** | **0.98816** |

## 2. Robustness and fragility

`correct` = the verdict matched what the operation should produce.
Photographs are reported separately, since synthetic extremes are there to
probe the limits rather than to represent typical input.

| operation | should be | correct (photos) | correct (all) | mean BER | mean blocks flagged |
|---|---|---|---|---|---|
| none (control) | authentic | 8/8 | 10/10 | 0.0000 | 0.0 |
| PNG re-save | authentic | 8/8 | 10/10 | 0.0000 | 0.0 |
| JPEG q95 | authentic | 7/8 | 9/10 | 0.0011 | 0.1 |
| JPEG q90 | authentic | 7/8 | 9/10 | 0.0020 | 0.2 |
| JPEG q80 | authentic | 7/8 | 9/10 | 0.0035 | 0.6 |
| JPEG q70 | authentic | 6/8 | 7/10 | 0.0043 | 1.2 |
| JPEG q60 | authentic | 3/8 | 4/10 | 0.0060 | 1.8 |
| JPEG q50 | authentic | 5/8 | 6/10 | 0.0088 | 3.2 |
| noise sigma=1 | authentic | 8/8 | 10/10 | 0.0000 | 0.0 |
| noise sigma=3 | authentic | 7/8 | 9/10 | 0.0000 | 0.0 |
| noise sigma=5 | authentic | 7/8 | 9/10 | 0.0002 | 0.1 |
| resize 0.75x | authentic | 7/8 | 8/10 | 0.0000 | 0.0 |
| resize 0.5x | authentic | 4/8 | 4/10 | 0.0000 | 0.0 |
| brightness +1 | authentic | 3/8 | 5/10 | 0.0000 | 0.0 |
| brightness +5 | authentic | 3/8 | 5/10 | 0.0001 | 0.0 |
| contrast x1.01 | authentic | 5/8 | 7/10 | 0.0000 | 0.0 |
| splice / copy-move | tampered | 8/8 | 10/10 | 0.0109 | 5.7 |
| blank region | tampered | 8/8 | 10/10 | 0.0111 | 6.5 |
| single 32x32 edit | tampered | 7/8 | 9/10 | 0.0025 | 1.3 |

## 3. Localisation accuracy

Flagged blocks vs the region actually edited.

| edit | precision | recall | F1 |
|---|---|---|---|
| splice / copy-move | 1.000 | 0.375 | 0.528 |
| blank region | 1.000 | 0.416 | 0.574 |
| single 32x32 edit | 1.000 | 1.000 | 1.000 |

## 4. Threshold sweep

Current settings: LSB_BLOCK_TAMPER_RATIO = 0.45, MIN_TAMPER_BLOCKS = 1.

Detection = content-altering edits correctly flagged. 
False alarms = content-preserving operations wrongly flagged.

| ratio | min blocks | detection | false alarms |
|---|---|---|---|
| 0.1 | 1 | 30/30 | 22/160 |
| 0.1 | 2 | 25/30 | 15/160 |
| 0.1 | 3 | 21/30 | 14/160 |
| 0.1 | 4 | 21/30 | 14/160 |
| 0.1 | 6 | 20/30 | 10/160 |
| 0.1 | 8 | 17/30 | 8/160 |
| 0.1 | 12 | 9/30 | 7/160 |
| 0.2 | 1 | 30/30 | 22/160 |
| 0.2 | 2 | 25/30 | 15/160 |
| 0.2 | 3 | 21/30 | 14/160 |
| 0.2 | 4 | 21/30 | 14/160 |
| 0.2 | 6 | 20/30 | 10/160 |
| 0.2 | 8 | 17/30 | 8/160 |
| 0.2 | 12 | 9/30 | 7/160 |
| 0.25 | 1 | 29/30 | 9/160 |
| 0.25 | 2 | 23/30 | 6/160 |
| 0.25 | 3 | 18/30 | 5/160 |
| 0.25 | 4 | 17/30 | 5/160 |
| 0.25 | 6 | 10/30 | 5/160 |
| 0.25 | 8 | 6/30 | 4/160 |
| 0.25 | 12 | 1/30 | 4/160 |
| 0.3 | 1 | 29/30 | 9/160 |
| 0.3 | 2 | 23/30 | 6/160 |
| 0.3 | 3 | 18/30 | 5/160 |
| 0.3 | 4 | 17/30 | 5/160 |
| 0.3 | 6 | 10/30 | 5/160 |
| 0.3 | 8 | 6/30 | 4/160 |
| 0.3 | 12 | 1/30 | 4/160 |
| 0.35 | 1 | 29/30 | 9/160 |
| 0.35 | 2 | 23/30 | 6/160 |
| 0.35 | 3 | 18/30 | 5/160 |
| 0.35 | 4 | 17/30 | 5/160 |
| 0.35 | 6 | 10/30 | 5/160 |
| 0.35 | 8 | 6/30 | 4/160 |
| 0.35 | 12 | 1/30 | 4/160 |
| 0.4 | 1 | 29/30 | 9/160 |
| 0.4 | 2 | 23/30 | 6/160 |
| 0.4 | 3 | 18/30 | 5/160 |
| 0.4 | 4 | 17/30 | 5/160 |
| 0.4 | 6 | 10/30 | 5/160 |
| 0.4 | 8 | 6/30 | 4/160 |
| 0.4 | 12 | 1/30 | 4/160 |
| 0.45 | 1 | 29/30 | 9/160 |  <- current
| 0.45 | 2 | 23/30 | 6/160 |
| 0.45 | 3 | 18/30 | 5/160 |
| 0.45 | 4 | 17/30 | 5/160 |
| 0.45 | 6 | 10/30 | 5/160 |
| 0.45 | 8 | 6/30 | 4/160 |
| 0.45 | 12 | 1/30 | 4/160 |
| 0.5 | 1 | 24/30 | 3/160 |
| 0.5 | 2 | 16/30 | 3/160 |
| 0.5 | 3 | 9/30 | 2/160 |
| 0.5 | 4 | 3/30 | 2/160 |
| 0.5 | 6 | 0/30 | 1/160 |
| 0.5 | 8 | 0/30 | 0/160 |
| 0.5 | 12 | 0/30 | 0/160 |
| 0.55 | 1 | 24/30 | 3/160 |
| 0.55 | 2 | 16/30 | 3/160 |
| 0.55 | 3 | 9/30 | 2/160 |
| 0.55 | 4 | 3/30 | 2/160 |
| 0.55 | 6 | 0/30 | 1/160 |
| 0.55 | 8 | 0/30 | 0/160 |
| 0.55 | 12 | 0/30 | 0/160 |
| 0.6 | 1 | 24/30 | 3/160 |
| 0.6 | 2 | 16/30 | 3/160 |
| 0.6 | 3 | 9/30 | 2/160 |
| 0.6 | 4 | 3/30 | 2/160 |
| 0.6 | 6 | 0/30 | 1/160 |
| 0.6 | 8 | 0/30 | 0/160 |
| 0.6 | 12 | 0/30 | 0/160 |
| 0.65 | 1 | 24/30 | 3/160 |
| 0.65 | 2 | 16/30 | 3/160 |
| 0.65 | 3 | 9/30 | 2/160 |
| 0.65 | 4 | 3/30 | 2/160 |
| 0.65 | 6 | 0/30 | 1/160 |
| 0.65 | 8 | 0/30 | 0/160 |
| 0.65 | 12 | 0/30 | 0/160 |