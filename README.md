# Cross-View Embeddings: Boulder County

Pre-computed cross-view embeddings for 3,303 points of interest in Boulder County, Colorado. Each POI has a learned embedding that encodes both its aerial appearance (from NAIP 30cm imagery) and its street-level appearance (from Mapillary) into a shared vector space.

This means you can take an aerial image of a building and find its matching street-level view, or vice versa, using simple cosine similarity.

Read the background: Future LinkedIn post

## What's in the box

```
data/
  embeddings_256d.jsonl     # INT8 quantized cross-view embeddings (256-d, 256 bytes/POI)
  embeddings_128d.jsonl     # INT8 quantized cross-view embeddings (128-d, 128 bytes/POI)
  embeddings_64d.jsonl      # INT8 quantized cross-view embeddings (64-d, 64 bytes/POI)
  poi_metadata.json         # Full POI metadata (names, categories, coordinates, building IDs)
  naip_chips/               # 384x384 NAIP aerial RGB chips (.npz)
notebooks/
  cross_view_retrieval.ipynb  # Demo: aerial-to-ground and ground-to-aerial retrieval
  explore_embeddings.ipynb    # Clustering, visualization, spatial analysis
```

## Embedding format

Each line in the `.jsonl` files is a JSON object:

```json
{
  "poi_gers_id": "f7476c3d-c480-4b8a-a0c5-e066605def16",
  "name": "Boulder County Carpet Care",
  "ugl_vector": [23, -104, 57, ...],
  "ugl_dim": 256,
  "ugl_scale": 127.0,
  "facade_bearing": 111.0,
  "entrance_lat": 40.0905094,
  "entrance_lon": -105.3461074
}
```

The `ugl_vector` is an INT8-quantized L2-normalized embedding. To use it:

```python
import numpy as np

def load_embeddings(path):
    """Load INT8 embeddings and dequantize to float."""
    embeddings, metadata = [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            vec = np.array(entry['ugl_vector'], dtype=np.float32) / entry['ugl_scale']
            vec = vec / np.linalg.norm(vec)  # Re-normalize after dequantization
            embeddings.append(vec)
            metadata.append(entry)
    return np.stack(embeddings), metadata
```

## Retrieval

Cross-view retrieval is cosine similarity between embeddings:

```python
# Find the 5 nearest street-level matches for an aerial query
sims = embeddings @ query_vector
top5 = np.argsort(-sims)[:5]
```

## Retrieval performance

Evaluated on held-out ground-aerial pairs (5,569 pairs across 2,258 locations):

| Embedding Dim | Storage per POI | Recall@1 | Recall@5 | INT8 Quantization Fidelity |
|:---:|:---:|:---:|:---:|:---:|
| 512-d (full) | 512 B | 90.7% | 99.6% | 0.999+ |
| 256-d | 256 B | ~91% | ~99% | 0.999+ |
| 128-d | 128 B | 91.1% | ~99% | 0.999+ |
| 64-d | 64 B | 90.2% | ~98% | 0.999+ |

## Data sources

- Aerial imagery: [NAIP](https://naip-usdaonline.hub.arcgis.com/) 2023 Colorado, 30cm resolution (public domain, USDA)
- Street-level imagery: [Mapillary](https://www.mapillary.com/) (CC BY-SA 4.0). Image IDs are included in `poi_metadata.json` for fetching via the [Mapillary API](https://www.mapillary.com/developer/api-documentation).
- POI and building data: [Overture Maps](https://overturemaps.org/) (ODbL)

## Viewing street-level images

Each POI includes Mapillary image IDs. You can view any image at:

```
https://www.mapillary.com/app/?pKey={image_id}
```

Or fetch programmatically via the Mapillary API with a [client token](https://www.mapillary.com/developer).

## License

- Embeddings and code: MIT
- Aerial chips: public domain (USDA NAIP)
- POI metadata derived from Overture Maps (ODbL) and Mapillary (CC BY-SA 4.0)

## About

Built by [Zephyr](https://zephyr.com). See also: [cross-view-embeddings](https://github.com/zephr-xyz/cross-view-embeddings) (private, model training code).
