# Bridging the Gap Between What Satellites See and What You See on the Street

Every map you've ever used has a blind spot.

Satellite and aerial imagery gives us a god's-eye view of the world: rooftops, parking lots, building footprints. Street-level imagery shows us what we actually experience: storefronts, entrances, signage, the texture of a neighborhood. These two perspectives describe the same places but in fundamentally different ways, and until now, there hasn't been a good way to connect them computationally.

At Zephyr we've been working on cross-view embeddings: learned representations that encode both aerial and street-level views of a location into a shared mathematical space. A building's rooftop and its facade become points near each other in this space, despite looking nothing alike in raw pixels.

This is harder than it sounds. You're asking a model to understand that a rectangular roof seen from 1,000 feet up corresponds to a two-story brick building with a glass storefront seen from across the street. It's a 90-degree perspective shift with almost zero visual overlap. Traditional computer vision approaches that work within a single viewpoint break down entirely here.

## Why This Matters

Single-perspective embeddings are everywhere. They power visual search, place recognition, image retrieval. But they're limited to matching like with like: aerial to aerial, street to street. That constraint excludes some of the most valuable geospatial problems.

Predicting what you'll see before you get there. If you can align aerial and ground perspectives in a shared space, you can infer street-level characteristics from overhead imagery alone. What does this building's entrance look like? Is there signage? What's the facade material? You can answer these questions for locations where no one has ever taken a street photo, using only satellite coverage.

Closing the last-mile navigation gap. Your map says you've arrived. But where's the door? POI pins are typically placed at building centroids, which can be 10-50 meters from the actual entrance. Cross-view embeddings allow you to snap a location to the correct facade and approach point by understanding the geometric relationship between overhead footprints and ground-level access points.

Detecting change without ground truth. When the aerial embedding of a location diverges from its historical street-level embedding, something has changed: a building demolished, a storefront renovated, a new structure. You get a change detection signal without needing fresh street-level imagery.

## The Bigger Picture: Geospatial Data Meets World Models

There's a reason this work feels timely. The AI world is moving toward "world models," systems that build internal representations of how the physical world works, not just how it looks in a single image. Cross-view embeddings are a step in that direction for geospatial understanding.

Traditional geospatial data is rich but flat. A building footprint tells you the shape and location of a structure. A POI record tells you the name and category of a business. Satellite imagery tells you about land cover and roof materials. None of these representations alone capture what it's like to *be at* a place: what you'd see approaching from the north, where you'd walk to find the entrance, what the neighborhood feels like at street level.

Cross-view embeddings start to bridge this gap. By learning a shared representation across viewpoints, you're encoding something closer to a geometric understanding of place. Not just "what does this look like from above" or "what does this look like from the street" but "what *is* this place, structurally, from any angle."

This is a fundamentally different kind of geospatial primitive. Instead of discrete data types (polygons, points, rasters) that each describe one facet of a location, you get a dense vector that compresses multi-perspective understanding into a single representation. That vector can be compared, clustered, searched, and used as input to downstream models, including the kind of embodied reasoning systems that need to understand the 3D world from limited observations.

## What We've Learned So Far

A few things surprised us in this work.

Geometry matters more than pixels. Encoding spatial relationships (the bearing between a camera and a facade, the shape of a building footprint, the offset from centroid to entrance) dramatically improves alignment quality. Raw visual similarity across viewpoints is weak; geometric context makes it tractable.

Coverage compounds. We have access to street-level imagery for millions of locations. But cross-view embeddings make that coverage multiplicative: every street-level observation enriches our understanding of the corresponding aerial view, and vice versa. Locations with both perspectives improve our model's ability to reason about locations with only one.

Small embeddings go far. Through careful representation learning, we can compress cross-view understanding into remarkably compact vectors that still retrieve accurately. The table below shows retrieval performance across embedding sizes, evaluated on a held-out set of 5,569 ground-aerial pairs spanning 2,258 locations.

| Embedding Dim | Storage per POI | Recall@1 | Recall@5 | INT8 Quantization Fidelity |
|:---:|:---:|:---:|:---:|:---:|
| 512-d (full) | 512 B | 90.7% | 99.6% | 0.999+ |
| 256-d | 256 B | ~91% | ~99% | 0.999+ |
| 128-d | 128 B | 91.1% | ~99% | 0.999+ |
| 64-d | 64 B | 90.2% | ~98% | 0.999+ |

A few things stand out. Performance barely degrades as you shrink the embedding from 512 dimensions down to 64. The 128-d vector actually slightly outperforms the full 512-d one, suggesting the higher dimensions mostly capture noise at this dataset scale. And INT8 quantization (halving storage again) preserves cosine fidelity above 0.999, meaning you can store a cross-view embedding for a location in 64 bytes and still get near-perfect retrieval. That matters for real-world deployment where you need embeddings cached on-device for millions of POIs.

## Where This Goes

We think cross-view embeddings are a foundational primitive for next-generation geospatial AI, not an end product but a building block. The ability to reason across viewpoints opens the door to:

- Navigation agents that understand the physical world, not just a graph of road segments
- Geospatial foundation models pre-trained on multi-perspective data
- Autonomous systems that can match what they see against overhead maps in real time
- Richer POI understanding that goes beyond a name and a pin

We're early in this work and there's a lot still to figure out. But the core insight feels durable: the world looks different from every angle, and the models that can reconcile those differences will understand places in ways that single-perspective systems never will.

---

*We're building this at Zephyr. If you're working on geospatial AI, embodied navigation, or multi-view representation learning, I'd love to connect.*
