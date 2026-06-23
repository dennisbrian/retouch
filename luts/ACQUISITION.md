# Film Stock LUT Acquisition

The Retouch Engine ships a small set of **demo** `.cube` LUTs in this directory.
They exist so the grading pipeline has working examples and so that you can
sanity-test the LUT application stage end-to-end. They are **NOT** real film
stock profiles.

## Why Demo LUTs?

Professional film stock LUTs (RNI, VSCO, Dehancer) are commercial products.
Each `.cube` file represents hundreds of hours of measurement, color science,
and artist grading against a specific stock. Their licenses generally forbid
redistribution, even alongside the engine that loads them.

We cannot ship Kodak Portra 400 or Fuji 400H profiles in this repository
without exposing the project to IP risk. The four demo LUTs shipped here
are algorithmically generated, safe to redistribute, and clearly labelled.
They demonstrate the LUT application machinery end-to-end; they are not
photographic look-a-likes.

## Recommended Film Stock Libraries

### 1. RNI Films (Really Nice Images) — recommended
- URL: <https://reallyniceimages.com>
- Price: ~$99 - $199 per pack
- ~50 film stocks including Kodak Portra 160/400/800, Fuji Pro 400H,
  Cinestill 800T, Polaroid, and slide film
- Designed for stills photography; Adobe `.cube` format
- High quality and well-documented

### 2. VSCO Film
- URL: <https://vsco.co/films>
- Price: ~$29 - $129 per pack
- 30+ stocks, popular in wedding and portrait photography
- Includes Fuji 400H, Kodak Portra, Kodak Gold, Ilford HP5

### 3. Fujifilm X-Trans Film Simulation Profiles — free for Fuji owners
- Available to owners of Fuji X-Trans sensor cameras
- Includes Velvia, Astia, Provia, Classic Chrome, Classic Neg, Eterna
- Distributed as `.cube` profiles via Fujifilm X RAW STUDIO
- Free with camera ownership; check the Fujifilm global support site
  for the current download path

### 4. Dehancer
- URL: <https://dehancer.com>
- Price: ~$99 (Photoshop/Lightroom plugin); $199+ (standalone)
- ~60 film stocks including motion-picture stocks (Kodak Vision3 5219,
  Fuji Eterna Vivid) plus still stocks
- Includes grain and halation emulation beyond just colour grading

> Prices and pack contents change over time. Verify on the vendor's site
> before purchasing.

## How to Install

1. Purchase or download the `.cube` files from one of the sources above.
2. Copy them into this `luts/` directory.
3. Verify the engine sees them:
   ```python
   from retouch.lut import list_available_luts
   print(list_available_luts())
   ```
4. Reference the LUT by its stem (filename without `.cube`) in your recipe:
   ```json
   {
     "grading": {
       "lut": "Kodak_Portra_400",
       "lut_strength": 0.85
     }
   }
   ```

## License Reminder

The `.cube` files you drop into this directory retain the license of their
original source. **Do not commit commercial LUTs to this repository.** They
are personal or team assets and should be installed locally per-developer
or loaded from a private shared file server.

If you build a recipe that depends on a commercial LUT, document the source
in the recipe's metadata so other users know where to acquire it.
