# Change log — audited v2.1

## Baseline problems from the supplied run
- Training mIoU climbed to ~0.865 while validation mIoU stayed near ~0.33.
- Validation loss bottomed early and rose materially, consistent with overfitting and/or a preprocessing/distribution mismatch.
- Augmentation depended on optional Albumentations and photometric jitter could affect the full 4-channel RGBA tensor.
- Both encoders were trained from scratch.
- The separately uploaded checkpoint metadata/state keys belonged to a different MFNet architecture; it is not a valid resume point for this model.

## Patch changes
- `my_model.py`: two ImageNet-initialized ResNet-18 streams, one RGB and one thermal; multi-scale cross-modal attention; U-Net-like decoder and dropout. Thermal conv1 is adapted from pretrained RGB kernels.
- `dataset.py`: strict RGBA/thermal input check; paired spatial augmentations; RGB-only colour jitter and independent mild thermal intensity jitter; nearest-neighbour mask resizing; `_flip` manifest handling; train/validation canonical split-overlap check.
- `train.py`: lower LR for pretrained encoders, higher LR for fusion/decoder, CUDA AMP, gradient clipping, validation-driven LR reduction, early stopping, full resumable checkpoints with architecture checks, new output folders, all-class and foreground mIoU, per-class IoU and prediction plots.
- `audit_mfnet.py`: read-only audit for split overlap, duplicate entries, missing pairs, mask IDs, image/mask sizes, class imbalance and thermal-alpha statistics.

## Verification limits
- Python syntax compilation passed.
- Synthetic model forward/backward, one-epoch CPU end-to-end smoke test and same-architecture resume smoke test passed.
- Actual MFNet data audit and full CUDA training must be run on the lab PC. They were not performed in this sandbox.
- No future validation mIoU is promised or claimed.


## v2.1 review fixes
- Fixed a critical AMP-path bug: the scaled loss is now backpropagated exactly once per batch.
- Adapted the pretrained one-channel thermal stem using the mean of RGB kernels rather than their sum, to avoid unnecessarily amplifying activations when the thermal stream has its own normalization.
- Corrected the audit's thermal statistic: it now reports actual pixel-level mean/std, plus per-image mean ranges. The previous label incorrectly described the mean of per-image minima as a mean intensity.
- Audits overlap for train/validation, train/test and validation/test, inventories physical flip variants, and reports canonical image stems not present in split manifests.
- Bumped the architecture/checkpoint ID and output folders so the corrected run does not reuse v2 outputs.
