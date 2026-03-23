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
        Generate Grad-CAM heatmap
        Args:
            input_tensor: (1, 3, H, W)
            class_idx: Target class index
        Returns:
            Heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor, clinical_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        
        # Normalize
        cam = F.relu(cam)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay heatmap on original image"""
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        return overlay

def create_grad_cam_visualization(model, grad_cam, image_tensor, original_image, class_idx, clinical_tensor=None):
    """
    Create Grad-CAM visualization and return as base64 encoded image
    """
    try:
        # Generate Grad-CAM
        cam = grad_cam.generate(image_tensor, class_idx, clinical_tensor)
        
        # Convert original image to numpy
        image_np = np.array(original_image.resize((224, 224)))
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.stack([image_np]*3, axis=-1)
        
        # Create overlay
        overlay = grad_cam.overlay_heatmap(image_np, cam, alpha=0.4)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    
    except Exception as e:
        logger.error(f"Grad-CAM visualization error: {e}")
        return None
