# Third-party providers and assets

Dream.exe's base package does not vendor optional provider source, model
weights, simulator assets, datasets, or generated outputs. Installing or
preparing a provider does not grant permission to redistribute its code,
weights, training data, or assets. Review the exact upstream terms for every
revision and checkpoint before use or redistribution.

`integrations/download_checkpoints.sh` is an explicit acquisition client, not
a redistribution mechanism. It refuses license-gated packages until the caller
passes the corresponding acceptance ID and verifies every downloaded file
against the packaged `core.json` size and SHA-256.

This file is an engineering inventory, not legal advice.

The benchmark `pipeline-runtime.json` manifest and the packaged
`dvd_lora_shared.json` and `dvd_lora_specific.json` configuration files are
original Dream.exe project materials. They are distributed under this
repository's Apache-2.0 license; this does not change the separate terms of
the DVD source, base models, or checkpoint weights they reference.

| Provider/input | Reviewed source revision | Source terms | Model/asset boundary |
|---|---|---|---|
| DVD | `EnVision-Research/DVD@5501c8bbf5983554f5bc8d3747a26e0d0c49d4ee` | Apache-2.0 in the reviewed checkout | The pinned `FayeHongfeiZhang/DVD@769534dec2d77f8da3667719568fbe57fe69cd9b` checkpoint declares CC BY-NC 4.0 and requires explicit acceptance. Dream.exe's fine-tuned LoRA is distributed separately. |
| Video Depth Anything | `DepthAnything/Video-Depth-Anything@4f5ae23172ba60fd7bc11ef671cca678842c7072` | Apache-2.0 in the reviewed checkout | Upstream documents Small (`vits`) weights as Apache-2.0 and Base/Large (`vitb`/`vitl`) weights as CC BY-NC 4.0. Dream.exe currently defaults to Large (`vitl`), so the default VDA checkpoint choice is non-commercial. |
| CoTracker | `facebookresearch/co-tracker@82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | CC BY-NC 4.0 in the reviewed checkout | The pinned `facebook/cotracker3@bf55ea50d4390e1820a267f131cd6587240fb2c5` checkpoint is a core external input with the same non-commercial restriction and requires explicit acceptance. |
| GroundingDINO | `IDEA-Research/GroundingDINO@856dde20aee659246248e20734ef9ba5214f5e44` | Apache-2.0 in the reviewed checkout | The checkpoint and configuration are external. Dream.exe ships only a manifest-attested two-line compatibility patch, not provider source or binaries. |
| SAM 2 | `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 in the reviewed checkout | Checkpoints and Hydra configuration selections are external inputs. |
| BERT base uncased | `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` | Apache-2.0 in the pinned model card | External GroundingDINO text-encoder files; downloaded but not redistributed by Dream.exe. |
| Wan2.1 DVD runtime files | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | Apache-2.0 in the pinned model card | The selected diffusion, VAE, and UMT5 files are external inputs consumed by DVD; they are not the optional Wan2.2 generator. |
| Wan2.2 video generation | `Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc` | Apache-2.0 in the reviewed checkout | `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e` weights are acquired separately and are never redistributed or downloaded by Dream.exe. Review the checkpoint/model-card terms before use or redistribution. |
| RoboSuite | PyPI `robosuite==1.5.2` | Review the installed distribution's notices | The code package does not vendor RoboSuite. |
| RoboCasa | `robocasa/robocasa@9a3a78680443734786c9784ab661413edb87067b` plus the manifest-attested 13-file task-success and five-file runtime-compatibility patch set | MIT for the pinned upstream source; RoboCasa assets are CC BY 4.0; the Dream.exe-owned patch files are Apache-2.0 | Source is fetched and patched as an external dependency. The benchmark data release contains only the exact frozen scene asset closure declared by `bench/sources/sim-runtime.json`; it does not contain the source episode dataset. A runtime shadow is an ignored caller-owned copy, not redistributed source authority. |

Pinned source links:

- [DVD](https://github.com/EnVision-Research/DVD/tree/5501c8bbf5983554f5bc8d3747a26e0d0c49d4ee)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything/tree/4f5ae23172ba60fd7bc11ef671cca678842c7072)
- [CoTracker](https://github.com/facebookresearch/co-tracker/tree/82e02e8029753ad4ef13cf06be7f4fc5facdda4d)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44)
- [SAM 2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc)

## Redistribution requirements

- Retain the upstream license and notices required for every redistributed
  patch or artifact.
- Record each checkpoint's source URL, immutable revision or digest, license,
  and redistribution terms.
- Do not use non-commercial providers or weights in commercial deployments.
- Keep the pinned RoboCasa source and episode datasets outside Dream.exe code
  and benchmark distributions. The benchmark contains only its declared frozen
  scene asset closure with the required attribution.
- Do not redistribute checkouts under `external/` unless the applicable
  upstream terms explicitly permit it.
