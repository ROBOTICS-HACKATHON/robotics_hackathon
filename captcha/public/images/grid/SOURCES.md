# Grid photo sources

Real photographs downloaded from Pexels and visually checked at the saved 600 × 600 crop. No AI image generation was used. Use is governed by the [Pexels License](https://www.pexels.com/license/); the photographs are not covered by the repository's code license.

## Training photos

The nine photographs below and the nine animals in [../SOURCES.md](../SOURCES.md) form the training pool. The first 160 image challenges use only this pool.

| Local file | Source photo |
| --- | --- |
| `car-1.avif` | [Gray car, photo 18481284](https://www.pexels.com/photo/sports-car-on-a-parking-lot-18481284/) |
| `car-2.avif` | [Red car, photo 32096056](https://www.pexels.com/photo/red-sports-car-parked-in-urban-park-setting-32096056/) |
| `car-3.avif` | [Parked sports cars, photo 5516079](https://www.pexels.com/photo/modern-sports-cars-parked-on-street-5516079/) |
| `bicycle-1.avif` | [Orange bicycle, photo 10810694](https://www.pexels.com/photo/a-bicycle-parked-near-the-door-10810694/) |
| `bicycle-2.avif` | [Red bicycle, photo 13246712](https://www.pexels.com/photo/red-bike-parked-beside-a-pole-13246712/) |
| `bicycle-3.avif` | [Parked bicycles, photo 19825197](https://www.pexels.com/photo/parked-urban-bikes-19825197/) |
| `flower-1.avif` | [Sunflower and roses, photo 9703886](https://www.pexels.com/photo/close-up-photo-of-sunflower-9703886/) |
| `flower-2.avif` | [Tulips and roses, photo 36842001](https://www.pexels.com/photo/colorful-tulip-and-rose-display-on-street-36842001/) |
| `flower-3.avif` | [Roses and tulips, photo 31727782](https://www.pexels.com/photo/vibrant-display-of-roses-and-tulips-in-bloom-31727782/) |

Downloads use `https://images.pexels.com/photos/ID/pexels-photo-ID.jpeg?auto=compress&cs=tinysrgb&w=600&h=600&fit=crop`.

The grids also reuse the nine animal photographs documented in [../SOURCES.md](../SOURCES.md). The bicycle parking photograph includes a background car and is not used as a negative tile in car challenges. Grids are fixed, with no timers or random replacement, so the robot can repeat each challenge.

## Held-out test photos

The final 40 image challenges use only these 18 different photographs. No source photo in this table is used by a training grid. Each saved crop was visually checked; unsuitable crops were replaced before defining the challenges. The split is attached to each challenge as `split: 'train' | 'test'`.

| Local file | Source photo |
| --- | --- |
| `test-cat-1.avif` | [Pexels photo 416160](https://www.pexels.com/photo/416160/) |
| `test-cat-2.avif` | [Pexels photo 1056251](https://www.pexels.com/photo/1056251/) |
| `test-cat-3.avif` | [Pexels photo 127028](https://www.pexels.com/photo/127028/) |
| `test-dog-1.avif` | [Pexels photo 37427157](https://www.pexels.com/photo/37427157/) |
| `test-dog-2.avif` | [Pexels photo 23849245](https://www.pexels.com/photo/23849245/) |
| `test-dog-3.avif` | [Pexels photo 16622787](https://www.pexels.com/photo/16622787/) |
| `test-bird-1.avif` | [Pexels photo 33683977](https://www.pexels.com/photo/33683977/) |
| `test-bird-2.avif` | [Pexels photo 19233293](https://www.pexels.com/photo/19233293/) |
| `test-bird-3.avif` | [Pexels photo 9814452](https://www.pexels.com/photo/9814452/) |
| `test-car-1.avif` | [Pexels photo 120049](https://www.pexels.com/photo/120049/) |
| `test-car-2.avif` | [Red sports car on a forest road, Pexels photo 210019](https://www.pexels.com/photo/210019/) |
| `test-car-3.avif` | [Pexels photo 30454878](https://www.pexels.com/photo/30454878/) |
| `test-bicycle-1.avif` | [Pexels photo 10785190](https://www.pexels.com/photo/10785190/) |
| `test-bicycle-2.avif` | [Pexels photo 5836909](https://www.pexels.com/photo/5836909/) |
| `test-bicycle-3.avif` | [Pexels photo 5312233](https://www.pexels.com/photo/5312233/) |
| `test-flower-1.avif` | [Pexels photo 736230](https://www.pexels.com/photo/736230/) |
| `test-flower-2.avif` | [Pexels photo 1444624](https://www.pexels.com/photo/1444624/) |
| `test-flower-3.avif` | [Pexels photo 8357451](https://www.pexels.com/photo/8357451/) |
