#!/usr/bin/env python3
"""
Export cross-view embeddings from trained model checkpoint to public JSONL format.

Reads the XVEE manifest + trained model checkpoint and produces INT8-quantized
embeddings at multiple MRL truncation dimensions (64, 128, 256).

Usage:
    python3 export_embeddings.py \
        --manifest /path/to/xvee_manifest.json \
        --checkpoint /path/to/checkpoint.pt \
        --ground-cache /path/to/ground_embeddings/ \
        --aerial-cache /path/to/aerial_embeddings/ \
        --output-dir ../data/

Requires: torch, numpy
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)


def quantize_int8(embedding):
    """Quantize L2-normalized float embedding to INT8."""
    scale = 127.0
    quantized = torch.clamp(torch.round(embedding * scale), -127, 127).to(torch.int8)
    return quantized, scale


def load_manifest(path):
    """Load XVEE manifest with POI metadata."""
    with open(path) as f:
        return json.load(f)


def export_dimension(embeddings, manifest, dim, output_path):
    """Export embeddings truncated to dim and quantized to INT8."""
    z_trunc = F.normalize(embeddings[:, :dim], dim=-1)
    z_int8, scale = quantize_int8(z_trunc)

    # Measure quantization fidelity
    z_deq = z_int8.float() / scale
    fidelity = F.cosine_similarity(z_trunc, z_deq, dim=-1).mean().item()

    count = 0
    with open(output_path, 'w') as f:
        for i, poi in enumerate(manifest):
            entry = {
                'poi_gers_id': poi['poi_gers_id'],
                'name': poi['name'],
                'ugl_vector': z_int8[i].tolist(),
                'ugl_dim': dim,
                'ugl_scale': scale,
                'facade_bearing': poi.get('facade_bearing'),
                'entrance_lat': poi.get('entrance_lat'),
                'entrance_lon': poi.get('entrance_lon'),
            }
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')
            count += 1

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path}: {count} entries, {dim}-d INT8, "
          f"fidelity={fidelity:.4f}, {size_kb:.0f} KB")


def export_metadata(manifest, output_path):
    """Export full POI metadata (no embeddings) as JSON."""
    pois = []
    for poi in manifest:
        pois.append({
            'poi_gers_id': poi['poi_gers_id'],
            'name': poi['name'],
            'category': poi.get('category'),
            'latitude': poi['latitude'],
            'longitude': poi['longitude'],
            'entrance_lat': poi.get('entrance_lat'),
            'entrance_lon': poi.get('entrance_lon'),
            'building_gers_id': poi.get('building_gers_id'),
            'building_source': poi.get('building_source'),
            'facade_bearing': poi.get('facade_bearing'),
            'facade_cardinal': poi.get('facade_cardinal'),
            'entrance_type': poi.get('entrance_type'),
            'venue_type': poi.get('venue_type'),
            'mapillary_ids': poi.get('mapillary_ids', []),
            'n_images': poi.get('n_images', len(poi.get('mapillary_ids', []))),
        })

    with open(output_path, 'w') as f:
        json.dump(pois, f, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path}: {len(pois)} POIs, {size_kb:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export cross-view embeddings")
    parser.add_argument('--manifest', required=True, help="Path to xvee_manifest.json")
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--ground-cache', required=True, help="Path to ground embedding cache")
    parser.add_argument('--aerial-cache', required=True, help="Path to aerial embedding cache")
    parser.add_argument('--output-dir', default='../data/', help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading manifest...")
    manifest = load_manifest(args.manifest)
    print(f"  {len(manifest)} POIs")

    print("\nExporting POI metadata...")
    export_metadata(manifest, os.path.join(args.output_dir, 'poi_metadata.json'))

    print("\nLoading checkpoint and computing embeddings...")
    print("  (This step requires the trained model and cached backbone features)")
    print("  If you don't have the model, use the pre-exported JSONL files in data/")

    # The actual model loading would go here. For reference:
    #
    #   checkpoint = torch.load(args.checkpoint, map_location='cpu')
    #   model = Geo2FineTune(...)
    #   model.load_state_dict(checkpoint['model_state_dict'])
    #   model.eval()
    #
    #   # Load cached backbone features for each POI
    #   ground_features = ...  # (N, 1024) from DINOv2
    #   geom_features = ...    # (N, 7) geometric features
    #
    #   with torch.no_grad():
    #       embeddings = model.encode_ground(ground_features, geom_features)
    #
    #   for dim in [64, 128, 256]:
    #       export_dimension(embeddings, manifest, dim,
    #                        os.path.join(args.output_dir, f'embeddings_{dim}d.jsonl'))

    print("\nDone. Embedding export requires model checkpoint (not included in public repo).")
    print("Pre-exported embeddings are available in the data/ directory.")


if __name__ == '__main__':
    main()
