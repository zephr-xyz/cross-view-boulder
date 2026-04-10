# Data

## Files

- `poi_metadata.json` - 3,303 POIs with names, categories, coordinates, building IDs, Mapillary image IDs
- `embeddings_256d.jsonl` - INT8 quantized cross-view embeddings, 256 dimensions (256 bytes/POI)
- `embeddings_128d.jsonl` - INT8 quantized cross-view embeddings, 128 dimensions (128 bytes/POI)
- `embeddings_64d.jsonl` - INT8 quantized cross-view embeddings, 64 dimensions (64 bytes/POI)
- `naip_chips/` - 384x384 NAIP aerial RGB chips as .npz files

## Generating embeddings

The `.jsonl` embedding files are exported from a trained model checkpoint using `scripts/export_embeddings.py`. The model and training code are in the private [cross-view-embeddings](https://github.com/zephr-xyz/cross-view-embeddings) repo.

## NAIP chips

Each `.npz` file contains:
- `rgb`: (3, 384, 384) uint8 array, RGB aerial imagery at 30cm/pixel
- `building_mask`: (729,) boolean array, patch-level building footprint mask

Source: USDA NAIP 2023 Colorado via AWS Open Data (public domain).
