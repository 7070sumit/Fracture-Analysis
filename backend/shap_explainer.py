import shap
import numpy as np
import torch
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
import warnings

# Suppress shap warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

logger = logging.getLogger(__name__)

def create_shap_visualization(model, image_tensor, clinical_tensor, feature_names=None):
    """
    Create SHAP visualization for clinical features and return as base64 encoded image
    """
    if feature_names is None:
        feature_names = ['Age_std', 'BMI_std', 'P02SEX', 'V00DIAB', 'V00FALL']
        
    try:
        model.eval()
        device = image_tensor.device
        
        # Clinical data as numpy array
        clin_np = clinical_tensor.cpu().numpy()
        
        # Create wrapper function that only takes clinical features
        def clinical_wrapper(clin_array):
            with torch.no_grad():
                # Convert to tensor
                clin_t = torch.tensor(clin_array, dtype=torch.float32).to(device)
                
                # Repeat image tensor for batch size
                batch_size = clin_array.shape[0]
                img_batch = image_tensor.repeat(batch_size, 1, 1, 1)
                
                # Forward pass
                logits = model(img_batch, clin_t)
                probs = torch.softmax(logits, dim=1)
                
                # Return probability of fracture (class 1)
                return probs[:, 1].cpu().numpy()
        
        # We need a background dataset for KernelExplainer.
        # A zero vector represents an "average" standardized patient.
        background = np.zeros((1, clin_np.shape[1]))
        
        # Initialize KernelExplainer
        # Turn off logging/progress bar for SHAP if possible
        explainer = shap.KernelExplainer(clinical_wrapper, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(clin_np, silent=True)
        
        # Generate visualization
        # Use a non-interactive backend
        plt.switch_backend('agg')
        plt.figure(figsize=(8, 4))
        
        # Using a horizontal bar chart
        shap_val = shap_values[0] # Single instance
        
        y_pos = np.arange(len(feature_names))
        colors = ['#ff0051' if val > 0 else '#008bfb' for val in shap_val]
        
        plt.barh(y_pos, shap_val, color=colors)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('SHAP Value (Impact on Fracture Risk)')
        plt.title('Clinical Feature Importance')
        
        # Add a vertical line at 0
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add values to bars
        max_abs_val = max(abs(val) for val in shap_val) if any(shap_val) else 0.1
        offset = max_abs_val * 0.05
        
        for i, v in enumerate(shap_val):
            if v > 0:
                plt.text(v + offset, i, f'+{v:.3f}', va='center', ha='left', color='#ff0051', fontweight='bold')
            elif v < 0:
                plt.text(v - offset, i, f'{v:.3f}', va='center', ha='right', color='#008bfb', fontweight='bold')
            else:
                plt.text(v, i, f'{v:.3f}', va='center', ha='center', color='gray')
                
        # Expand xlim slightly to accommodate text
        plt.xlim(min(shap_val) - offset * 4, max(shap_val) + offset * 4)
                
        # Invert y-axis to match typical SHAP plots
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        import traceback
        logger.error(f"SHAP visualization error: {e}")
        logger.error(traceback.format_exc())
        return None
