import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAMPlusPlusPyTorch:
    """
    Grad-CAM++ implementation for PyTorch models.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        b, k, u, v = gradients.size()
        
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * alpha_num + torch.sum(activations * gradients.pow(3), dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        weights = torch.max(gradients, torch.zeros_like(gradients))
        
        alphas_thresholding = torch.where(gradients != 0, alphas, torch.zeros_like(alphas))
        
        weights = torch.sum(alphas_thresholding * weights, dim=(2,3), keepdim=True)
        weights = F.relu(weights)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.view(u, v).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize 0-1
        
        if isinstance(class_idx, (int, np.integer)):
            final_class_idx = int(class_idx)
        else:
            final_class_idx = class_idx.item() if hasattr(class_idx, 'item') else int(class_idx)
            
        try:
            final_score = score.item()
        except AttributeError:
            final_score = float(score)
        
        return cam, final_class_idx, final_score


def overlay_heatmap(img_path, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image.
    Args:
        img_path (str): Path to original image.
        heatmap (np.array): 224x224 heatmap values [0, 1].
        alpha (float): Transparency factor.
    Returns:
        superimposed_img (np.array): RGB image with heatmap overlay.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap_color * alpha + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)
    
    return superimposed_img


def calculate_severity_score(heatmap, intensity_threshold=0.5):
    """
    Calculates a severity score (0-10) based on heatmap activation area and intensity.
    Formula: Score = (Mean Intensity of Active Region * Area of Active Region) * Scaling Factor
    """
    active_mask = heatmap > intensity_threshold
    
    if np.sum(active_mask) == 0:
        return 0.0
    
    area_ratio = np.sum(active_mask) / (224 * 224)
    mean_intensity = np.mean(heatmap[active_mask])
    
    raw_score = (area_ratio * 0.7 + mean_intensity * 0.3) * 100
    final_score = min(10.0, raw_score * 0.5)
    
    return round(final_score, 2)
