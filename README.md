# Fashion Color Palette Finder

Upload a photo (or series of photos) and this model identifies the clothing and accessories in the image, then extracts the dominant colors used — giving you a clean color palette straight from the outfit.

## Background

This project started as a take-home technical challenge for a startup interview — build a model that pulls a color palette out of an outfit photo. It sat untouched for a couple years after that, until I came back, cleaned up the code, added documentation, and deployed it as a real interactive app.

## How it works

1. **Segmentation** — uses a YOLO segmentation model (`yolo26n-seg.pt`) to detect and isolate clothing and accessories in the input image, separating them from the background and skin.
2. **Color extraction** — analyzes the segmented regions to extract the dominant colors present in the outfit.
3. **Output** — returns a color palette representing the primary colors used across the clothing and accessories in the image.

## Try it

🔗 [Link to deployed app] <!-- replace with your actual Gradio/hosted link -->

Or run it locally:

```bash
git clone https://github.com/VenPrograms/Fashion_Color_Palettes.git
cd Fashion_Color_Palettes
pip install -r requirements.txt   # add a requirements.txt if you don't have one yet
python main_without_threads.py
```

To launch the interactive Gradio interface instead:

```bash
cd gradio_folder
python app.py   # replace with your actual entry-point filename
```
