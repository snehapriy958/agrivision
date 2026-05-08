"""
utils/gradcam.py — Grad-CAM heatmap generator for PyTorch models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Usage:
        gradcam = GradCAM(model, target_layer=model.model.layer4)
        try:
            heatmap = gradcam.generate(tensor, class_idx=None)
        finally:
            gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self._activations : torch.Tensor | None = None
        self._gradients   : torch.Tensor | None = None
        self._hooks_active : bool = False

        self._forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)
        self._hooks_active  = True

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activations(
        self,
        _module : nn.Module,
        _input  : tuple,
        output  : torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self,
        _module  : nn.Module,
        _grad_in : tuple,
        grad_out : tuple,
    ) -> None:
        # grad_out[0]: gradient of the loss w.r.t. this layer's output tensor
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor : torch.Tensor,
        class_idx    : int | None = None,
    ) -> np.ndarray:
        """
        Produce a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : Preprocessed image tensor of shape (1, 3, H, W).
                           Must NOT be inside inference_mode or no_grad context.
            class_idx    : Class index to explain. Uses argmax if None.

        Returns:
            Float32 numpy array of shape (H, W) normalized to [0, 1].

        Raises:
            RuntimeError : If hooks were not registered or gradients are None/NaN.
        """
        if not self._hooks_active:
            raise RuntimeError(
                "GradCAM hooks have been removed. "
                "Create a new GradCAM instance for each prediction."
            )

        self.model.eval()

        # Reset captured state before each generate call
        self._activations = None
        self._gradients   = None

        self.model.zero_grad(set_to_none=True)

        # ── Forward pass ──────────────────────────────────────────────
        # No requires_grad on input needed — gradients flow through
        # layer4's trainable parameters and are captured by the hook.
        logits = self.model(input_tensor)          # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # ── Backward pass ─────────────────────────────────────────────
        score = logits[0, class_idx]
        score.backward()

        # ── Validate captures ─────────────────────────────────────────
        activations = self._activations   # (1, C, H, W)
        gradients   = self._gradients     # (1, C, H, W)

        if activations is None or gradients is None:
            raise RuntimeError(
                "Hooks did not capture activations or gradients. "
                "Ensure the model is not in inference_mode."
            )
        if torch.isnan(gradients).any():
            raise RuntimeError("Gradients contain NaN — check model numerics.")

        # ── Grad-CAM formula ──────────────────────────────────────────
        # Global average pool gradients over spatial dims → importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)              # (1, H, W)
        cam = F.relu(cam)                                      # discard negatives

        # ── Resize to input resolution ────────────────────────────────
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(1),                 # (1, 1, H, W) for interpolate
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()                           # (input_h, input_w)

        # ── Normalize to [0, 1] ───────────────────────────────────────
        cam_np  = cam.cpu().numpy()
        cam_min = cam_np.min()
        cam_max = cam_np.max()

        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # ── Sanity assertions ─────────────────────────────────────────
        assert cam_np.shape == (input_h, input_w), \
            f"Heatmap shape mismatch: expected {(input_h, input_w)}, got {cam_np.shape}"
        assert not np.isnan(cam_np).any(), \
            "Heatmap contains NaN values."
        assert 0.0 <= cam_np.min() and cam_np.max() <= 1.0, \
            f"Heatmap values out of [0, 1]: min={cam_np.min()}, max={cam_np.max()}"

        return cam_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Cleanup — called by the CALLER, not internally by generate()
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks. Safe to call multiple times."""
        if self._hooks_active:
            self._forward_hook.remove()
            self._backward_hook.remove()
            self._hooks_active = False