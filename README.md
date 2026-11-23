# Weave Sample Desk

Inspect woven samples in physical coordinates. Open the synthetic plum-and-indigo swatch or add face, transmitted, or raking-light photographs and register each to the same sample plane.

Start with `./start.sh` and open the address it prints. The default port is 19998. The named volume retains runs and downloaded evidence across container recreation. `./start.sh --down` stops this service.

Add photographs as JPEG, PNG, or TIFF. Select four corners clockwise from the top left. Mark polygons for observation areas, two points for a selvedge transect, or matching ordered points for motif offsets. Readings retain source photographs, calibration, selected marks, overlay image and numerical values. The evidence packet contains original photos.

The desk measures cloth tone, solid dye contrast, dyed area, dominant weave periods, transmitted openings, selvedge transition, motif offsets, warp/weft tone balance, paired usable coverage, surface spots, paired color difference and straight cut plans. The preparation routine provides independent input cleanup and routine photo transformations.

For a damaged edge, mark a transmitted-light patch around the cut-out, draw two straight cut lines (two or more ordered points each), choose the cloth side to keep and the clearance each line must keep from any visible opening. The cut plan rejects a line whose clearance band holds a visible opening or cloth the photograph never captured, then adopts at most one passing line — the one that leaves more cloth, with the larger opening margin breaking a tie. The adopted line is stored on the run; later Cloth tone readings on any registered photograph of the same sheet count only the kept visible cloth and exclude the cut-away side, transmitted openings and missing capture. A plain transmitted-openings reading is saved on its own and does not replace the adopted cut.

Lab and dye measurements describe photographed RGB, limited by lighting and profile accuracy. The warp/weft prediction uses equal-area linear RGB blending. Clipped or transparent pixels are excluded from comparisons. No measurement identifies thread chemistry or guarantees a physical fabric property.

Each run accepts at most 12 photographs, 100 marks and 160 readings. Input photographs must be at most 20 MiB and 14 megapixels. The calibrated image uses up to four pixels per millimetre and a maximum long edge of 1,300 pixels.
