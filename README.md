# Weave Sample Desk

Inspect woven samples in physical coordinates. Open the synthetic plum-and-indigo swatch or add face, transmitted, or raking-light photographs and register each to the same sample plane.

Start with `./start.sh` and open the address it prints. The default port is 19998. The named volume retains runs and downloaded evidence across container recreation. `./start.sh --down` stops this service.

Add photographs as JPEG, PNG, or TIFF. Select four corners clockwise from the top left. Mark polygons for observation areas, two points for a selvedge transect, or matching ordered points for motif offsets. Readings retain source photographs, calibration, selected marks, overlay image and numerical values. The evidence packet contains original photos.

The desk measures cloth tone, solid dye contrast, dyed area, dominant weave periods, transmitted openings, selvedge transition, motif offsets, warp/weft tone balance, paired usable coverage, surface spots and paired color difference. A dye reference check reads a known half-dyed patch through the same photograph's own undyed and dyed references and accepts it only inside the interval you set. One checked set can be adopted for the batch — the half-dyed reading closest to one half wins, ties go to the calmer undyed cloth — and dyed area can then be measured on the batch's other photographs against that adopted set. Saving further readings never replaces the adoption; only adopting again does. The preparation room provides independent input cleanup and routine photo transformations.

Lab and dye measurements describe photographed RGB, limited by lighting and profile accuracy. The warp/weft prediction uses equal-area linear RGB blending. Clipped or transparent pixels are excluded from comparisons. No measurement identifies thread chemistry or guarantees a physical fabric property.

Each run accepts at most 12 photographs, 100 marks and 160 readings. Input photographs must be at most 20 MiB and 14 megapixels. The calibrated image uses up to four pixels per millimetre and a maximum long edge of 1,300 pixels.
