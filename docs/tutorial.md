![[overview.png]]

### Getting started

Upload a metallography image (JPG, PNG, or BMP) using the uploader on the left. That's it — analysis starts automatically once it's in.

### Tabs

- **Original / Draw** — your original image. Draw a rectangle on it and hit "Calculate Selected Region Porosity" to check just that region.
- **Area Mask** — the red overlay highlights the selected area. Please adjust the mask so it fits within the object and cover all pores.

        The calculation and visualisation is only correct if the area mask is correctly adjusted (holes are coverred and black area are excluded)

- **Pores** — pores highlighted in blue. Tick "Highlight 3 Biggest Pores" to mark and measure the three largest ones.
- **Grain Grid** — only shows up for etched samples. Make sure Zoom Factor is set correctly first, since that's what turns pixels into real µm.

### Control Panels (left)

Please click Recalculate button to apply changes.

- **Sample Type**: Unetched Images: porosity only, Etched Images: also measure grain size.
- **Basic Settings**
  - OCR auto-scale: ticked if you have Tesseract OCR installed, to auto detect Zoom Factor
  - Fast Mask Processing: it downsizes the image resolution for faster calculation, with lower accuracy. The smaller processing size, the faster. Size applies for the longest side. Default is on, 512 pixel.
  - Zoom factor: manually given or used OCR detection. Default 5000 µm.

- **Mask Setting** — Use to adjust the area mask.
  - Adjust pore sizes: default assumes small pores, for bigger pores please adjust.
  - Adjust mask shrink: Fast Mask Processing often leads to unprecicion on the border, shrink the mask to get better border porosity and full porosity.
  - Crop legend: You can choose to pick either in pixel (default 100) or percentage of height (default 10%). The area counted from the bottom will be deduced from the mask. To have better control and precision, please use the area mask tab and draw it yourself.

- **Advanced Setting** — Only for advanced user. Morphology, border, and threshold controls in the algorithm. The advanced setting will override Pore Size and Mask Shrink adjustment!
- **Analysis Actions** — reset everything to default, or recalculate after changing something.
