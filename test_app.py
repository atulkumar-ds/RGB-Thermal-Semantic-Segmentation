"""Focused unit and integration tests for app.py and inference pipeline."""

import os
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from app import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    MFNET_COLORMAP,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    NUM_CLASSES,
    colorize_mask,
    create_overlay,
    inspect_checkpoint_file,
    load_model,
    preprocess_pair,
    run_inference,
)


class TestAppPipeline(unittest.TestCase):

    def test_inspect_checkpoint_real(self):
        exists, is_lfs, size, desc = inspect_checkpoint_file(DEFAULT_CHECKPOINT_PATH)
        self.assertTrue(exists, f"Checkpoint not found at {DEFAULT_CHECKPOINT_PATH}")
        self.assertFalse(is_lfs, "Expected real checkpoint, but detected LFS pointer")
        self.assertGreater(size, 300 * 1024 * 1024, "Checkpoint should be ~349 MB")

    def test_inspect_checkpoint_missing(self):
        exists, is_lfs, size, desc = inspect_checkpoint_file("non_existent_path.pt")
        self.assertFalse(exists)
        self.assertFalse(is_lfs)
        self.assertEqual(size, 0)

    def test_inspect_checkpoint_lfs_pointer(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".pt") as tmp:
            tmp.write(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:70f4adcb1349d31c5ec07dad0c3d507899222383b67ffbcd83376f094f104b55\n"
                "size 349118757\n"
            )
            tmp_path = tmp.name

        try:
            exists, is_lfs, size, desc = inspect_checkpoint_file(tmp_path)
            self.assertTrue(exists)
            self.assertTrue(is_lfs)
            self.assertIn("Git LFS pointer detected", desc)
        finally:
            os.remove(tmp_path)

    def test_preprocess_pair_dimensions_and_scaling(self):
        # Create non-native resolution images
        orig_w, orig_h = 800, 600
        rgb_img = Image.new("RGB", (orig_w, orig_h), color=(200, 100, 50))
        thermal_img = Image.new("L", (orig_w, orig_h), color=180)

        tensor, (ret_w, ret_h) = preprocess_pair(rgb_img, thermal_img)

        self.assertEqual(ret_w, orig_w)
        self.assertEqual(ret_h, orig_h)
        self.assertEqual(tensor.shape, (1, 4, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

        # Check channel values match expectations
        # RGB was (200/255, 100/255, 50/255)
        self.assertAlmostEqual(tensor[0, 0, 0, 0].item(), 200.0 / 255.0, places=2)
        self.assertAlmostEqual(tensor[0, 1, 0, 0].item(), 100.0 / 255.0, places=2)
        self.assertAlmostEqual(tensor[0, 2, 0, 0].item(), 50.0 / 255.0, places=2)
        # Thermal channel 3 was 180 / 255
        self.assertAlmostEqual(tensor[0, 3, 0, 0].item(), 180.0 / 255.0, places=2)

    def test_model_loading_and_inference(self):
        model, metadata = load_model(DEFAULT_CHECKPOINT_PATH, "cpu")
        self.assertIsNotNone(model)
        self.assertFalse(model.training)
        self.assertEqual(metadata.get("epoch"), 22)

        # Run forward inference with demo sample
        demo_rgb = Image.open("demo_samples/day_00001D_rgb.png")
        demo_thermal = Image.open("demo_samples/day_00001D_thermal.png")

        tensor, orig_dim = preprocess_pair(demo_rgb, demo_thermal)
        preds = run_inference(model, tensor, orig_dim, "cpu")

        self.assertEqual(preds.shape, (demo_rgb.height, demo_rgb.width))
        self.assertTrue(np.all(preds >= 0) and np.all(preds < NUM_CLASSES))

        # Check colorization
        colored = colorize_mask(preds)
        self.assertEqual(colored.shape, (demo_rgb.height, demo_rgb.width, 3))
        self.assertEqual(colored.dtype, np.uint8)

        # Check overlay
        overlay = create_overlay(demo_rgb, colored, alpha=0.5)
        self.assertEqual(overlay.size, demo_rgb.size)


if __name__ == "__main__":
    unittest.main()
