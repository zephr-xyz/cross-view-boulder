#!/usr/bin/env python3
"""
Build poi_metadata.json from the XVEE manifest.

This extracts the public-facing POI metadata (no embeddings, no internal fields)
from the training manifest and writes it to data/poi_metadata.json.

Usage:
    python3 build_poi_metadata.py --manifest /path/to/xvee_manifest.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='../data/poi_metadata.json')
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

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

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(pois, f, indent=2)

    print(f"Wrote {len(pois)} POIs to {args.output}")

    # Summary stats
    cats = {}
    for p in pois:
        c = p.get('category', 'unknown')
        cats[c] = cats.get(c, 0) + 1
    print(f"  {len(cats)} unique categories")
    print(f"  {sum(p['n_images'] for p in pois)} total Mapillary images")
    print(f"  Top categories: {', '.join(f'{c} ({n})' for c, n in sorted(cats.items(), key=lambda x: -x[1])[:5])}")


if __name__ == '__main__':
    main()
