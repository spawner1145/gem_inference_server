# Bundled inference source

These are ordinary source files committed with the server, not Git submodules.
No checkout or download of another source repository is required to run the server.

| Directory | Upstream | Revision / origin | License |
| --- | --- | --- | --- |
| `gem_x/gem`, `gem_x/configs` | NVlabs/GEM-X | `32992550dba114c62243fb55e361311972dce8f9` | `gem_x/LICENSE`, `gem_x/ATTRIBUTIONS.md` |
| `gem_x/third_party/soma/soma` | NVlabs/SOMA-X | `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5` | `gem_x/third_party/soma/LICENSE`, `ATTRIBUTIONS.MD` |
| `gem_x/third_party/sam-3d-body/sam_3d_body` | facebookresearch/sam-3d-body | `b5c765a0d89d789985e186d396315e7590887b94` | `gem_x/third_party/sam-3d-body/LICENSE` |
| `dinov3` | facebookresearch/dinov3 | Official Torch Hub source used by the validated local deployment | `dinov3/LICENSE.md` |

Only inference packages, configuration and attribution/license files are bundled.
Original copyright notices are retained. The two DINOv3 loaders in ViTPose and
SAM-3D-Body are adapted to load `gem_server.paths.DINOV3_SOURCE` with
`torch.hub.load(..., source="local")`. See the server's `ATTRIBUTIONS.md`.

Model weights and SOMA body assets remain separate downloads in `models/` and
are not part of this source distribution.
