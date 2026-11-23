# Weave Sample Desk

Inspect woven samples in physical coordinates. Open the synthetic plum-and-indigo swatch or add face, transmitted, or raking-light photographs and register each to the same sample plane.

Start with `./start.sh` and open the address it prints. The default port is 19998. The named volume retains runs and downloaded evidence across container recreation. `./start.sh --down` stops this service.

Add photographs as JPEG, PNG, or TIFF. Select four corners clockwise from the top left. Mark polygons for observation areas, two points for a selvedge transect, or matching ordered points for motif offsets. Readings retain source photographs, calibration, selected marks, overlay image and numerical values. The evidence packet contains original photos.

The desk measures cloth tone, solid dye contrast, dyed area, dominant weave periods, transmitted openings, selvedge transition, motif offsets, warp/weft tone balance, paired usable coverage, surface spots, paired color difference and raised-fiber comparison. The raised-fiber reading registers one face-light photograph against one raking-light photograph of the same patch. Mark one confirmed fuzz patch, one weave-sheen patch and the inspection area on the reference photograph. It scores two detection rules only where both photographs are clear: mark what both lights lift above its surroundings, and mark where the raking light lifts more than the face light. Enter the least fraction of confirmed fuzz that must stay marked and the most sheen that may be mis-marked; only one rule is adopted per reading. When both rules qualify, the rule retaining more confirmed fuzz is kept, then the one with fewer sheen false marks; when neither qualifies, the rule closest to both limits is kept. The saved report records the rule, both scores and the gates.

Open any saved reading and click its report image. The desk reports the matching position in each saved source photograph, the brightness read there, and which photograph is not clear at that point (transparent, unlit, clipped or outside the frame). Each photograph can be opened with that point circled, and the evidence packet still carries the original photos.

The point inspection uses the recorded source photographs and their calibration, so it shows the exact pixels the reading compared. The preparation room provides independent input cleanup and routine photo transformations.

Lab and dye measurements describe photographed RGB, limited by lighting and profile accuracy. The warp/weft prediction uses equal-area linear RGB blending. Clipped or transparent pixels are excluded from comparisons. No measurement identifies thread chemistry or guarantees a physical fabric property.

Each run accepts at most 12 photographs, 100 marks and 160 readings. Input photographs must be at most 20 MiB and 14 megapixels. The calibrated image uses up to four pixels per millimetre and a maximum long edge of 1,300 pixels.
