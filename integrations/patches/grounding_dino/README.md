# GroundingDINO compatibility delta

Dream.exe does not vendor GroundingDINO. The external-source manifest pins the
official checkout and the setup command applies the narrow patch declared in
`manifest.json`.

The patch replaces the deprecated Tensor `type()` dispatch argument with
`scalar_type()` in the two CUDA deformable-attention dispatch sites. This is
required to compile the pinned provider against the validated PyTorch 2.7.1
ABI; it does not change the algorithm.

The checkout, compiled extension, checkpoints, and GroundingDINO configuration
remain external runtime inputs and are never included in Dream.exe archives.
