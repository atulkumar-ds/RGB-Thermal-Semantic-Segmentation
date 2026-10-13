# Deep Audit Report — Before and After Changes

## A. Evidence from the supplied run

The supplied epoch-level graphs show:
- Training loss falls from roughly 2 to 0.32.
- Validation loss bottoms out early (around epochs 9–10) and then rises above 2 by epoch 50.
- Training mIoU reaches roughly 0.86, while validation mIoU plateaus around 0.33.
- Validation class IoUs at epoch 50 vary considerably: Unlabeled about 0.951, Car 0.543, Person 0.601, Bike 0.311, Curve 0.167, Car Stop 0.097, Guardrail 0.003, Color Cone 0.249, Bump 0.071.
- Prediction images show fragmented object regions and substantial misses compared with ground truth.

This pattern is strong evidence of overfitting and/or a data/distribution/preprocessing issue. The graphs alone cannot prove which single cause is dominant.

## B. Static code findings before changes

1. **Augmentation was optional and could silently be absent.** `dataset.py` only created augmentations when Albumentations imported; its fallback merely resized. The supplied requirements did not include Albumentations, so the no-augmentation fallback was a realistic path.
2. **Photometric RGB transforms were applied to the entire 4-channel RGBA array.** That means the alpha-encoded thermal values could be altered by the RGB brightness/contrast operation.
3. **Missing thermal data was silently replaced with zeroes.** The old loader accepted grayscale and RGB-only images and created a zero thermal channel, which could silently invalidate the two-modality experiment.
4. **Both encoders were randomly initialized.** The supplied custom model had two convolutional encoders, GroupNorm/SiLU blocks and five fusion scales, but no pretrained initialization. It had about 3 million trainable parameters and was trained from scratch on a small dataset.
5. **Overfitting controls were limited.** The supplied trainer used AdamW and cosine decay but had no early stopping or validation-triggered LR reduction.
6. **Metrics could conceal foreground quality.** mIoU included class 0 (`Unlabeled`). This is a valid convention in some RGB–thermal implementations, but classwise/foreground reporting is necessary. The revised trainer reports both all-class and foreground mIoU.
7. **The separately uploaded `best_checkpoint.pt` did not belong to the supplied current model.** Its metadata said `ARCH: mfnet`, its epoch was 35, and its state keys (for example `conv1_rgb...`) did not match `CustomDualStreamUNet` state names. It should not be resumed in this model.
8. **Run bookkeeping was easy to mix.** The training artifacts included model outputs and checkpoints from distinct training histories. The new model uses a new versioned checkpoint/plot directory to avoid overwriting old experiments.

## C. Research comparison

The most directly verified recent MFNet result among sources checked was SpatialSeg: its official repository reports 64.6% mIoU with Swin-B and supplies pretrained weights. It requires a preprocessed dataset layout with separate RGB, Thermal and Label folders and uses a separate pipeline. CMX's official repository reports 58.2% for MiT-B2 and 59.7% for MiT-B4. SGFN's comparison table reports 59.0% for MiT-B2 and 59.9% for MiT-B4. CMNeXt is reported around 59.9% in comparison literature; EAEFNet's repository reports 58.91%. RTFNet is a relevant dual-encoder RGB–thermal baseline; its README warns that its metrics differ from MFNet's and include the unlabelled class.

Sources:
- https://github.com/lab-sun/SpatialSeg
- https://github.com/huaaaliu/RGBX_Semantic_Segmentation
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11053727/
- https://www.researchgate.net/publication/368935656_Delivering_Arbitrary-Modal_Semantic_Segmentation
- https://github.com/FreeformRobotics/EAEFNet
- https://github.com/yuxiangsun/RTFNet

These reported results use each paper/repository's own dataset preparation and evaluation protocol; they do not guarantee the same result on this project's validation split. The updated custom model is a practical transfer-learning upgrade rather than a reproduction of the highest reported architecture.

## D. Changes in v2

1. A pretrained dual-stream ResNet-18 feature pyramid replaces the two random convolutional encoders. Thermal first-convolution weights are adapted from the pretrained RGB kernels.
2. Multi-scale cross-modal channel/spatial attention and a U-Net-like decoder are retained as the custom fusion architecture.
3. RGB and thermal normalization are handled per modality within the model.
4. Synchronized horizontal flip and scale/crop/pad augmentation now always runs for training. RGB color jitter is applied only to RGB; thermal jitter is independent and mild.
5. The dataset now rejects images without a true 4-channel RGB+thermal input rather than silently zero-filling thermal.
6. The trainer uses a lower LR for pretrained encoders and higher LR for fusion/decoder, optional CUDA AMP, gradient clipping, a ReduceLROnPlateau scheduler, and early stopping.
7. The trainer saves all-class mIoU and foreground mIoU, per-class IoU, predictions, training history, and checkpoints under a new v2 folder.
8. Checkpoint loading checks an architecture ID and refuses incompatible files; RNG tensors are moved back to CPU where required during restore.

## E. Post-change verification

Verification performed in this environment:
- Python syntax compilation passed for the updated `dataset.py`, `my_model.py`, `train.py`, and `audit_mfnet.py`.
- Synthetic model smoke test with `pretrained=False`: input `[1,4,64,96]` returned logits `[1,9,64,96]`; a cross-entropy backward pass completed.
- Read-only dataset audit passed on a generated toy RGBA dataset: split-overlap detection, RGBA channel inspection, mask pixel counts and label validation all ran.
- A one-epoch end-to-end CPU smoke run on synthetic data completed; it wrote predictions, plots, `last_checkpoint.pt`, and `best_checkpoint.pt`.
- Resume smoke test from the generated `last_checkpoint.pt` resumed at epoch 2 and completed successfully. This verifies the new same-architecture resume path.
- The pretrained download path, mixed-precision CUDA path, and full run were not exercised in this sandbox. The user's actual MFNet files are not attached here, so real mask IDs, split leakage, RGBA alpha statistics, and paired sizes still must be verified locally by running `audit_mfnet.py`.
- No full MFNet training was performed here, and no new validation mIoU is claimed.


## F. v2.1 corrective review after audit screenshot
The user's local audit screenshot showed no missing train/val pairs, no train/val canonical filename overlap, grayscale masks with IDs 0..8, and RGBA 480x640 inputs. However, v2.0's alpha statistic was mislabeled: it averaged per-image minima instead of computing the actual pixel mean. The revised audit now computes global pixel mean/std and pairwise overlap for all available splits.

A second post-change source review caught a critical issue not exercised by the original CPU-only synthetic test: the CUDA AMP branch called `scaler.scale(loss).backward()` twice on the same graph. v2.1 corrects this to one backward pass. A targeted one-epoch synthetic training test with a scaler-compatible test double then passed through that branch, alongside forward/backward and corrected audit-statistics tests. This does not substitute for a real CUDA AMP test. Do not launch full CUDA training with the earlier v2 patch; replace it with v2.1 and repeat the one-epoch integration test first.
