#!/usr/bin/env python3
"""
Build public cross-view embedding export from XVEE v7 neural feature cache.

Merges INT8 embeddings from the v7 model with POI metadata from the dataset
manifest to produce the public JSONL format for cross-view-boulder.

Usage:
    python3 build_v7_export.py \
        --cache-256 /path/to/neural_feature_cache_256d.jsonl \
        --cache-512 /path/to/neural_feature_cache_512d.jsonl \
        --manifest /path/to/xvee_dataset_v6.json \
        --output-dir ../data/
"""

import argparse
import json
import os
import sys


def load_cache(path):
    """Load neural feature cache JSONL into dict keyed by poi_gers_id."""
    cache = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            cache[entry['poi_gers_id']] = entry
    return cache


def load_manifest(path):
    """Load dataset manifest and build POI lookup."""
    with open(path) as f:
        samples = json.load(f)
    lookup = {}
    for s in samples:
        lookup[s['poi_gers_id']] = s
    return lookup


def export_embeddings(cache, manifest, dim, output_path, vector_key='aerial_ugl'):
    """Export embeddings in public JSONL format."""
    count = 0
    with open(output_path, 'w') as f:
        for poi_id, entry in sorted(cache.items()):
            meta = manifest.get(poi_id, {})
            gf = meta.get('geom_features', {})

            record = {
                'poi_gers_id': poi_id,
                'name': meta.get('name', ''),
                'ugl_vector': entry[vector_key],
                'ugl_dim': dim,
                'ugl_scale': entry['scale'],
                'facade_bearing': meta.get('facade_bearing'),
                'entrance_lat': meta.get('entrance_lat'),
                'entrance_lon': meta.get('entrance_lon'),
            }
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
            count += 1

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path}: {count} POIs, {dim}-d, {size_kb:.0f} KB")
    return count


def export_metadata(cache, manifest, output_path):
    """Export POI metadata JSON."""
    pois = []
    for poi_id in sorted(cache.keys()):
        meta = manifest.get(poi_id, {})
        pois.append({
            'poi_gers_id': poi_id,
            'name': meta.get('name', ''),
            'category': meta.get('category'),
            'latitude': meta.get('latitude', meta.get('entrance_lat')),
            'longitude': meta.get('longitude', meta.get('entrance_lon')),
            'entrance_lat': meta.get('entrance_lat'),
            'entrance_lon': meta.get('entrance_lon'),
            'building_gers_id': meta.get('building_gers_id'),
            'facade_bearing': meta.get('facade_bearing'),
            'mapillary_ids': [img['image_id'] for img in meta.get('ranked_images', [])],
            'n_images': len(meta.get('ranked_images', [])),
        })

    with open(output_path, 'w') as f:
        json.dump(pois, f, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path}: {len(pois)} POIs, {size_kb:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description="Build v7 public embedding export")
    parser.add_argument('--cache-256', required=True)
    parser.add_argument('--cache-512', default=None)
    parser.add_argument('--manifest', required=True,
                        help='xvee_dataset_v6.json manifest')
    parser.add_argument('--output-dir', default='../data/')
    parser.add_argument('--vector-key', default='aerial_ugl',
                        choices=['aerial_ugl', 'ground_ugl'],
                        help='Which embedding to export (default: aerial)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading v7 neural feature cache...")
    cache_256 = load_cache(args.cache_256)
    print(f"  256-d: {len(cache_256)} POIs")

    if args.cache_512:
        cache_512 = load_cache(args.cache_512)
        print(f"  512-d: {len(cache_512)} POIs")

    print("\nLoading dataset manifest...")
    manifest = load_manifest(args.manifest)
    print(f"  {len(manifest)} POIs in manifest")

    # Check overlap
    matched = sum(1 for pid in cache_256 if pid in manifest)
    print(f"  {matched}/{len(cache_256)} cache entries matched to manifest")

    print("\nExporting aerial embeddings (256-d)...")
    export_embeddings(cache_256, manifest, 256,
                      os.path.join(args.output_dir, 'embeddings_256d.jsonl'),
                      vector_key=args.vector_key)

    # Also export 128-d and 64-d by truncating the 256-d vectors
    print("\nExporting truncated embeddings (128-d, 64-d)...")
    count = 0
    for dim in [128, 64]:
        output_path = os.path.join(args.output_dir, f'embeddings_{dim}d.jsonl')
        with open(output_path, 'w') as f:
            for poi_id, entry in sorted(cache_256.items()):
                meta = manifest.get(poi_id, {})
                # Truncate vector to dim
                vec = entry[args.vector_key][:dim]
                record = {
                    'poi_gers_id': poi_id,
                    'name': meta.get('name', ''),
                    'ugl_vector': vec,
                    'ugl_dim': dim,
                    'ugl_scale': entry['scale'],
                    'facade_bearing': meta.get('facade_bearing'),
                    'entrance_lat': meta.get('entrance_lat'),
                    'entrance_lon': meta.get('entrance_lon'),
                }
                f.write(json.dumps(record, separators=(',', ':')) + '\n')
                count += 1
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  {output_path}: {count // (3 - [128, 64].index(dim))} POIs, {dim}-d, {size_kb:.0f} KB")
        count = 0

    print("\nExporting POI metadata...")
    export_metadata(cache_256, manifest, os.path.join(args.output_dir, 'poi_metadata.json'))

    print("\nDone.")


if __name__ == '__main__':
    main()
