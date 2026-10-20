# Upstream projects

- [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X), Apache-2.0. Inference follows its SOMA demo, preprocessing, model configuration and postprocessing. Tested source revision: `32992550dba114c62243fb55e361311972dce8f9`.
- [NVIDIA SOMA-X](https://github.com/NVlabs/SOMA-X), Apache-2.0. GEM-X submodule revision: `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`. The motion adapter uses its joint-orientation / forward-kinematics convention. Official resources are downloaded separately from `nvidia/soma-x` at the version listed in the model manifest.
- [Meta SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body), SAM License. GEM-X submodule revision: `b5c765a0d89d789985e186d396315e7590887b94`. Its original license applies to this dependency and its weights.
- [Meta DINOv3](https://github.com/facebookresearch/dinov3), DINOv3 License. Inference source is bundled in `vendor/dinov3` and loaded via local Torch Hub. See its bundled `LICENSE.md`.
- YOLOX (Apache-2.0), ByteTrack (MIT), and OpenMMLab HumanArt detector. Detection and tracking use GEM-X's self-contained adapter.

Model files, source dependencies and other assets are not relicensed by the server's Apache-2.0 license. Consult their original model cards / LICENSE files for usage and redistribution terms. No weights are included in the source distribution.

The overall server / client organization is inspired by the workspace's Kimodo API bridge. This client speaks the GEM service API and imports SOMA motion matrices rather than calling the Kimodo model.

## Bundled source layout

`vendor/gem_x` contains GEM inference packages/configs and the SOMA / SAM-3D-Body packages at the revisions listed above, with their original licenses and attribution files. `vendor/dinov3` contains the DINOv3 Python package and Hub entry point. No Git metadata, model weights, demos, or datasets are included in this source bundle.

Local integration changes: the two DINOv3 Hub loads in GEM ViTPose and SAM-3D-Body now use the bundled `vendor/dinov3` path with `source="local"`. Their model architecture and checkpoint loading behavior are otherwise retained.
