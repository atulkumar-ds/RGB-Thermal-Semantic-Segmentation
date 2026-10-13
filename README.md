# MFNet RGB + Thermal Segmentation — audited v2.1

This patch upgrades the existing custom RGB–thermal segmentation project without replacing its overall training workflow.

## What changed

- `my_model.py`: custom dual-stream ResNet-18 U-Net with multi-scale cross-modal channel/spatial attention. Both encoders start from ImageNet-pretrained weights by default. The thermal first convolution is adapted to one channel.
- `dataset.py`: paired geometry augmentation is implemented directly, so training does not silently lose augmentation when Albumentations is absent. RGB photometric jitter affects RGB only; thermal gain/bias jitter is independent and mild. Images lacking a real 4-channel RGBA input now fail clearly rather than silently substituting a zero thermal channel.
- `train.py`: separate lower learning rate for pretrained encoders, higher learning rate for fusion/decoder, CUDA AMP, gradient clipping, validation-driven LR reduction, early stopping, resumable checkpoints, foreground-only mIoU alongside 9-class mIoU, less frequent synchronous logging, and separated output folders.
- `audit_mfnet.py`: read-only data audit for split leakage, missing pairs, image/mask sizes, mask values, class imbalance and thermal alpha statistics.
- New outputs go under `checkpoints/resnet18_crossmodal_attention_v2_1/` and `plots/resnet18_crossmodal_attention_v2_1/`. Existing results are not overwritten.

## Environment setup (Windows / lab PC)

Activate the project's virtual environment first. Install the correct CUDA PyTorch and torchvision build from the official PyTorch selector/index for your driver. For an NVIDIA driver reporting CUDA 12.8 compatibility, this command is a starting point if your installed Python version has matching wheels:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining requirements:

```powershell
python -m pip install -r requirements.txt
```

Verify CUDA and torchvision:

```powershell
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

## Step 1 — Audit your actual dataset before training

From the repository root:

```powershell
python audit_mfnet.py --data-root ..\MFNet_dataset --image-check-limit 0
```

Review any warnings about missing files, train/validation overlap, unexpected labels, non-RGBA images, or a near-constant alpha channel. The audit script is read-only and does not download or alter data.

The loader now intentionally rejects RGB-only files because this model requires actual RGB + thermal inputs. If the downloaded dataset uses a different format, fix the data interpretation explicitly instead of silently training an RGB-only model.

## Step 2 — One-epoch integration test

```powershell
python train.py --epochs 1 --batch-size 4 --num-workers 4 --log-every 20 --vis-every 1
```

The first run may download the ImageNet ResNet-18 weights to the torchvision cache. This requires internet access once. If the machine cannot access that URL, `--no-pretrained` is available for a smoke test only; it is not the recommended final setting.

## Step 3 — Main training run

After the one-epoch test completes and you inspect the predictions:

```powershell
python train.py --epochs 60 --batch-size 4 --num-workers 4 --patience 12 --vis-every 5 --log-every 20
```

Use batch size 2 if CUDA runs out of memory. With the RTX 2000 Ada reported on the lab PC, batch size 4 with AMP is a reasonable starting point, but actual free VRAM and speed decide the final value.

Early stopping monitors validation mIoU. The run can stop before 60 epochs if validation has stalled for 12 epochs; that is intentional given the previous overfitting curve.

## Checkpoints and plots

- `checkpoints/resnet18_crossmodal_attention_v2_1/last_checkpoint.pt`: full resumable checkpoint, saved at each completed epoch.
- `checkpoints/resnet18_crossmodal_attention_v2_1/best_checkpoint.pt`: full checkpoint from the best validation mIoU epoch.
- `checkpoints/resnet18_crossmodal_attention_v2_1/model_epoch_NNN.pt`: smaller model-only snapshot every 10 epochs.
- `plots/resnet18_crossmodal_attention_v2_1/epoch_loss.png`
- `plots/resnet18_crossmodal_attention_v2_1/epoch_metrics.png`
- `plots/resnet18_crossmodal_attention_v2_1/miou_foreground.png`
- `plots/resnet18_crossmodal_attention_v2_1/per_class_iou.png`
- `plots/resnet18_crossmodal_attention_v2_1/learning_rate.png`
- `plots/resnet18_crossmodal_attention_v2_1/epoch_NNN_predictions.png`
- `plots/resnet18_crossmodal_attention_v2_1/training_history.json`

Resume only from this new architecture's checkpoint:

```powershell
python train.py --epochs 60 --resume checkpoints/resnet18_crossmodal_attention_v2_1/last_checkpoint.pt
```

Do not use the previously uploaded root-level `checkpoints/best_checkpoint.pt`: its metadata and parameter names identify a different MFNet architecture. The architecture check in this version refuses an incompatible checkpoint.

## Metrics

- `mIoU`: mean IoU across the 9 dataset IDs, retaining the existing convention that includes class 0 (`Unlabeled`).
- `foreground mIoU`: mean IoU across classes 1–8; displayed separately to expose foreground quality.
- `per_class_iou.png`: per-class validation IoU at the best validation mIoU epoch.

Keep both measures when comparing to papers. MFNet benchmark protocols, train/validation/test splits, and background/unlabelled handling vary between implementations.

## Literature context (published figures, not a promise for this run)

- The recent SpatialSeg repository reports 64.6% MFNet mIoU with Swin-B (and 59.4%/62.1% with Swin-T/S). It provides pretrained weights, but expects a preprocessed dataset with separate RGB, Thermal and Label folders and uses its own training pipeline: https://github.com/lab-sun/SpatialSeg
- CMX's official repository reports 58.2% with MiT-B2 and 59.7% with MiT-B4: https://github.com/huaaaliu/RGBX_Semantic_Segmentation
- SGFN's comparison reports 59.0% with MiT-B2 and 59.9% with MiT-B4: https://pmc.ncbi.nlm.nih.gov/articles/PMC11053727/
- CMNeXt is reported around 59.9% on MFNet in comparison literature: https://www.researchgate.net/publication/368935656_Delivering_Arbitrary-Modal_Semantic_Segmentation
- EAEFNet's repository reports 58.91% on MFNet: https://github.com/FreeformRobotics/EAEFNet
- RTFNet is a classic RGB–thermal two-encoder/one-decoder baseline; its official README explicitly notes that its metrics differ from MFNet's and include the unlabelled class: https://github.com/yuxiangsun/RTFNet

If the sole goal is to reproduce a top published MFNet result, SpatialSeg with its Swin-B configuration is the strongest directly verified result in the sources checked. That is a larger research-reproduction task and requires their preprocessed dataset format. This patch instead improves the current custom pipeline with pretrained ResNet-18 encoders while preserving the packed RGBA loader. Figures from papers/repos are not guaranteed or directly comparable to this project's custom validation split.
