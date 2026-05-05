import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self._activations : torch.Tensor | None = None
        self._gradients   : torch.Tensor | None = None

        self._forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)
        self._hooks_registered = True

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activations(
        self,
        _module : nn.Module,
        _input  : tuple,
        output  : torch.Tensor,
    ) -> None:
        self._activations = output

    def _save_gradients(
        self,
        _module   : nn.Module,
        _grad_in  : tuple,
        grad_out  : tuple,
    ) -> None:
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor : torch.Tensor,
        class_idx    : int | None = None,
    ) -> np.ndarray:
        self.model.eval()
        self._activations = None
        self._gradients   = None
        input_tensor = input_tensor.requires_grad_(True)


        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        logits = self.model(input_tensor)                # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        activations = self._activations   # (1, C, H, W)
        gradients   = self._gradients     # (1, C, H, W)

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations / gradients.")
        if torch.isnan(gradients).any():
            raise RuntimeError("Gradients contain NaN values.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * activations).sum(dim=1)          # (1, H, W)
        cam     = F.relu(cam)                                  # (1, H, W)

        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()                                            # (H, W)

        cam_np = cam.detach().cpu().numpy()
        cam_min = cam_np.min()
        cam_max = cam_np.max()

        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        assert cam_np.shape == (input_h, input_w), "Heatmap shape mismatch."
        assert not np.isnan(cam_np).any(),          "Heatmap contains NaN."
        assert cam_np.max() <= 1.0 and cam_np.min() >= 0.0, "Heatmap out of [0, 1]."

        try:
            return cam_np.astype(np.float32)
        finally:
            self.remove_hooks()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        if hasattr(self, "_hooks_registered") and self._hooks_registered:
            self._forward_hook.remove()
            self._backward_hook.remove()
            self._hooks_registered = False
        