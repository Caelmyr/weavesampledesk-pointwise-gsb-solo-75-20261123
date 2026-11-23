# Weave Sample Desk

Inspect woven samples in physical coordinates. Open the synthetic plum-and-indigo swatch or add face, transmitted, or raking-light photographs and register each to the same sample plane.

Start with `./start.sh` and open the address it prints. The default port is 19998. The named volume retains runs and downloaded evidence across container recreation. `./start.sh --down` stops this service.

Add photographs as JPEG, PNG, or TIFF. Select four corners clockwise from the top left. Mark polygons for observation areas or a seam window, two points for a selvedge transect, or matching ordered points for motif offsets. Readings retain source photographs, calibration, selected marks, overlay image and numerical values. The evidence packet contains original photos.

The desk measures cloth tone, solid dye contrast, dyed area, dominant weave periods, transmitted openings, selvedge transition, motif offsets, warp/weft tone balance, paired usable coverage, surface spots and paired color difference. Placement match chooses between two translation-only placements from ordered motif centres and a per-centre tolerance: only placements whose centres all fall within tolerance are eligible, with a recorded reason for every choice (closest worst-centre alignment, then the most seam area visible in both photos, then lowest RMS residual). Seam coverage then reuses a saved placement and classifies a marked seam window into pixels seen by both photos, only the reference, only the comparison, or neither. Saved readings—including the adopted placement—and original photographs are immutable snapshots, so later motif-offset readings never alter them. The preparation room provides independent input cleanup and routine photo transformations.

Lab and dye measurements describe photographed RGB, limited by lighting and profile accuracy. The warp/weft prediction uses equal-area linear RGB blending. Clipped or transparent pixels are excluded from comparisons. No measurement identifies thread chemistry or guarantees a physical fabric property.

Each run accepts at most 12 photographs, 100 marks and 160 readings. Input photographs must be at most 20 MiB and 14 megapixels. The calibrated image uses up to four pixels per millimetre and a maximum long edge of 1,300 pixels.
