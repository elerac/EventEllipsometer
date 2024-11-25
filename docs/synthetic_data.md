# Synthetic Data Generation

## Rendering Images with Mitsuba3

To render the synthetic data, [Mitsuba3](https://github.com/mitsuba-renderer/mitsuba3) renderer with '*_spectral_polarized' variant is required. 

The scene files (3D model, pbsdf) need to be prepared in the `scenes` directory. 

- [bunny_uv.ply](https://www.dropbox.com/scl/fi/jv8ncz9jrej1vc22ar9d2/20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG.zip?rlkey=l8nczue360jrq3o5y79k9nzjh&e=1&dl=0)
  - As the original model does not have the UV mapping, the UV-mapped model is provided at [here](https://blenderartists.org/t/uv-unwrapped-stanford-bunny-happy-spring-equinox/1101297).
  - I convert this model to the PLY format by using Blender.
- [pbsdf](https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html)

```bash
scenes
├── meshes
│   ├── bunny_uv.ply
│   └── ...
├── pbsdf
│   ├── 9_blue_billard_mitsuba
│   │   ├── 9_blue_billard_inpainted.pbsdf
│   │   └── ...
│   └── ...
└── ...
```

The rendering can be done by running the `render_images.py` script.

```bash
python scripts/render_images.py
```

## Converting Images to Event Data with DVS-Voltmeter

After render the images, the images are converted to the event data by the [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter) [Lin+, ECCV2022]. This repository self-contains the DVS-Voltmeter code that is modified for this project.

```bash
python scripts/imgs2event.py
```

## Mask 

If you want to add the mask to the event data, you first prepare a mask image and run the `delete_masked_events.py` script.
