import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int, clinical_tensor: torch.Tensor = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap for improved localization.
        
        Uses second-order gradient weighting (Grad-CAM++) for tighter spatial 
        localization compared to vanilla Grad-CAM.
        """
        self.model.eval()
        
        input_tensor = input_tensor.detach().requires_grad_(True)
        output = self.model(input_tensor, clinical_tensor)
        
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        gradients = self.gradients[0]      # (C, h, w)
        activations = self.activations[0]  # (C, h, w)
        
        # Grad-CAM++ weighting
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        sum_act = activations.sum(dim=(1, 2), keepdim=True)
        
        eps = 1e-8
        alpha = grad_2 / (2.0 * grad_2 + sum_act * grad_3 + eps)
        alpha = alpha * F.relu(gradients)
        weights = alpha.sum(dim=(1, 2))
        
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        
        return cam.cpu().detach().numpy()
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on original image (legacy, kept for compatibility)"""
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        return overlay


def _build_anatomy_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Build a mask that isolates bone/tissue regions in an X-ray and suppresses
    background, edge artifacts, labels, and non-anatomical bright objects
    (splints, blankets, collimation edges).
    
    Strategy:
      1. Convert to grayscale.
      2. Apply adaptive thresholding to find bright foreground.
      3. Find the largest connected component (= the body/bone region).
      4. Erode edges to avoid activating on border artifacts.
      5. Smooth to create soft transitions.
    
    Returns:
        Float mask (H, W) in [0, 1].
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    h, w = gray.shape
    
    # ── Step 1: Adaptive threshold to find all bright regions ───────────
    # Use a block-based adaptive threshold which handles varying brightness
    block_size = max(51, (min(h, w) // 10) | 1)  # must be odd
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -10
    )
    
    # Also apply a global threshold to catch overall bright regions
    thresh_val = max(15, int(gray.max() * 0.10))
    _, global_binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Combine: pixel must be bright in both
    combined = cv2.bitwise_and(binary, global_binary)
    
    # ── Step 2: Morphological cleanup ──────────────────────────────────
    # Close small gaps, then open to remove small noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
    
    # ── Step 3: Keep only the largest connected component (body region) ─
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        # Skip label 0 (background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        body_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        body_mask = cleaned
    
    # ── Step 4: Erode edges to pull mask away from image borders ───────
    # This prevents activation on collimation edges, splints at borders
    edge_margin = max(10, min(h, w) // 50)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_margin, edge_margin))
    eroded = cv2.erode(body_mask, k_erode, iterations=1)
    
    # Also explicitly zero out a border strip
    border = max(5, min(h, w) // 80)
    eroded[:border, :] = 0
    eroded[-border:, :] = 0
    eroded[:, :border] = 0
    eroded[:, -border:] = 0
    
    # ── Step 5: Gaussian blur for soft edges ───────────────────────────
    blur_k = max(31, (min(h, w) // 20) | 1)
    mask = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, (blur_k, blur_k), 0)
    
    return mask


def create_grad_cam_visualization(model, grad_cam, image_tensor, original_image, class_idx, clinical_tensor=None):
    """
    Bake the Grad-CAM heatmap directly onto the original image at full resolution.
    
    Pipeline:
    1. Generate Grad-CAM++ heatmap from the model.
    2. Upscale the small feature-map-sized heatmap to original image resolution.
    3. Apply an anatomy mask to suppress activation on non-bone regions
       (background, labels, splints, blankets, collimation edges).
    4. Smooth and blend with the original image using per-pixel alpha.
    """
    try:
        cam = grad_cam.generate(image_tensor, class_idx, clinical_tensor)
        
        orig_w, orig_h = original_image.size  # PIL: (width, height)
        image_np = np.array(original_image)
        
        # Convert to BGR for OpenCV
        if len(image_np.shape) == 2:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Upscale heatmap to original resolution
        heatmap_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply anatomy mask to constrain heatmap to bone/tissue regions
        anatomy_mask = _build_anatomy_mask(image_np)
        heatmap_masked = heatmap_resized * anatomy_mask
        
        # Re-normalize so strongest on-anatomy activation = 1.0
        hm_max = heatmap_masked.max()
        if hm_max > 0:
            heatmap_masked = heatmap_masked / hm_max
        
        # Smooth for clean edges
        blur_k = max(11, (min(orig_w, orig_h) // 30) | 1)
        heatmap_smooth = cv2.GaussianBlur(heatmap_masked, (blur_k, blur_k), 0)
        
        # Apply JET colormap
        heatmap_uint8 = (heatmap_smooth * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Per-pixel alpha blending (threshold to keep low-activation areas clean)
        threshold = 0.15
        alpha_mask = np.clip((heatmap_smooth - threshold) / (1.0 - threshold), 0, 1)
        alpha_mask = (alpha_mask * 0.85).astype(np.float32)
        alpha_3ch = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)
        
        # Blend
        image_f = image_bgr.astype(np.float32)
        heatmap_f = heatmap_colored.astype(np.float32)
        composited = (image_f * (1.0 - alpha_3ch) + heatmap_f * alpha_3ch).astype(np.uint8)
        
        # Encode to JPEG base64
        _, buffer = cv2.imencode('.jpg', composited, [cv2.IMWRITE_JPEG_QUALITY, 93])
        img_str = base64.b64encode(buffer).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    
    except Exception as e:
        import traceback
        logger.error(f"Grad-CAM visualization error: {e}")
        logger.error(traceback.format_exc())
        return None
