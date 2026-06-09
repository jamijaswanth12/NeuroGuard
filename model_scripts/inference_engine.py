import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gradcam_plus_plus


# --- PyTorch Model Definitions ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DenseNet169_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet169_CBAM, self).__init__()
        densenet = models.densenet169(pretrained=False)
        self.features = densenet.features
        self.cbam = CBAM(1664)
        self.classifier = nn.Sequential(
            nn.Linear(1664, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = self.cbam(features)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class InferenceEngine:
    def __init__(self):
        print("\n" + "=" * 50)
        print("  INFERENCE ENGINE  v5.0 (PyTorch-Only)")
        print("=" * 50 + "\n")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Portable paths: model_scripts/ -> parent (NeuroGuard_GitHub/) -> Models/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mri_model_path = os.path.join(base_dir, "Models", "mri_best_model_v2.pth")
        self.ct_model_path  = os.path.join(base_dir, "Models", "ct_best_model.pth")

        self.ct_model  = None
        self.mri_model = None
        self._load_models()

        self.mri_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_models(self):
        print("Loading models...")
        if os.path.exists(self.ct_model_path):
            try:
                self.ct_model = DenseNet169_CBAM(num_classes=3).to(self.device)
                sd = torch.load(self.ct_model_path, map_location=self.device)
                new_sd = {k.replace('base_model.', ''): v for k, v in sd.items()}
                self.ct_model.load_state_dict(new_sd, strict=False)
                self.ct_model.eval()
                print("CT Model loaded successfully.")
            except Exception as e:
                print(f"Critical Error loading CT model: {e}")
        else:
            print(f"CT Model not found at {self.ct_model_path}")

        if os.path.exists(self.mri_model_path):
            try:
                self.mri_model = DenseNet169_CBAM(num_classes=3).to(self.device)
                sd = torch.load(self.mri_model_path, map_location=self.device)
                new_sd = {k.replace('base_model.', ''): v for k, v in sd.items()}
                self.mri_model.load_state_dict(new_sd, strict=False)
                self.mri_model.eval()
                print("MRI Model loaded successfully.")
            except Exception as e:
                print(f"Critical Error loading MRI model: {e}")
        else:
            print(f"MRI Model not found at {self.mri_model_path}. See README for download.")

    def validate_modality(self, image_path, expected_type):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "Invalid image file format."
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            total_pixels = img.shape[0] * img.shape[1]
            hist_norm = hist / total_pixels
            ratio_255 = hist_norm[255][0]
            is_ct_likely = ratio_255 > 0.04
            predicted_type = 'CT' if is_ct_likely else 'MRI'
            if predicted_type != expected_type:
                return False, (
                    f"Warning: The uploaded image appears to be a {predicted_type} scan "
                    f"but you selected {expected_type} analysis. Please verify your file."
                )
            return True, "Valid"
        except Exception as e:
            print(f"Modality validation error: {e}")
            return True, "Modality check skipped"

    def _preprocess(self, image_path):
        img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
        img = self.mri_transform(img_pil)
        return img.unsqueeze(0).to(self.device)

    def predict(self, image_path, modality):
        is_valid, msg = self.validate_modality(image_path, modality)
        if not is_valid:
            return {'error': msg}
        if modality == 'CT':
            return self._predict_ct(image_path)
        elif modality == 'MRI':
            return self._predict_mri(image_path)
        else:
            return {'error': 'Unknown modality'}

    def _predict_ct(self, image_path):
        if self.ct_model is None:
            return {'error': 'CT Model not loaded. Check server logs.'}
        try:
            input_tensor = self._preprocess(image_path)
            with torch.no_grad():
                preds = self.ct_model(input_tensor)
                prob = torch.softmax(preds, dim=1)
                class_idx = torch.argmax(prob, dim=1).item()
                confidence = prob[0][class_idx].item()
            classes = ['Hemorrhagic', 'Ischemic', 'Normal']
            prediction = classes[class_idx]
            heatmap = np.zeros((224, 224))
            overlay = cv2.imread(image_path)
            severity = 0.0
            try:
                target_layer = self.ct_model.cbam
                gradcam = gradcam_plus_plus.GradCAMPlusPlusPyTorch(self.ct_model, target_layer)
                heatmap, _, _ = gradcam(input_tensor, class_idx=torch.tensor(class_idx))
                overlay = gradcam_plus_plus.overlay_heatmap(image_path, heatmap)
                severity = gradcam_plus_plus.calculate_severity_score(heatmap) if prediction != 'Normal' else 0.0
            except Exception as cam_err:
                print(f"Warning: GradCAM failed for CT: {cam_err}")
            if overlay is not None and overlay.shape[:2] != (224, 224):
                overlay = cv2.resize(overlay, (224, 224))
            return {'prediction': prediction, 'confidence': confidence, 'severity': severity, 'heatmap': heatmap, 'overlay': overlay}
        except Exception as e:
            return {'error': str(e)}

    def _predict_mri(self, image_path):
        if self.mri_model is None:
            return {'error': 'MRI Model not loaded. Check server logs.'}
        try:
            input_tensor = self._preprocess(image_path)
            with torch.no_grad():
                outputs = self.mri_model(input_tensor)
                prob = torch.softmax(outputs, dim=1)
                class_idx = torch.argmax(prob, dim=1).item()
                confidence = prob[0][class_idx].item()
            classes = ['Hemorrhagic', 'Ischemic', 'Normal']
            prediction = classes[class_idx]
            heatmap = np.zeros((224, 224))
            overlay = cv2.imread(image_path)
            severity = 0.0
            try:
                target_layer = self.mri_model.cbam
                gradcam = gradcam_plus_plus.GradCAMPlusPlusPyTorch(self.mri_model, target_layer)
                heatmap, _, _ = gradcam(input_tensor, class_idx=torch.tensor(class_idx))
                overlay = gradcam_plus_plus.overlay_heatmap(image_path, heatmap)
                severity = gradcam_plus_plus.calculate_severity_score(heatmap) if prediction != 'Normal' else 0.0
            except Exception as cam_err:
                print(f"Warning: GradCAM failed for MRI: {cam_err}")
            if overlay is not None and overlay.shape[:2] != (224, 224):
                overlay = cv2.resize(overlay, (224, 224))
            return {'prediction': prediction, 'confidence': confidence, 'severity': severity, 'heatmap': heatmap, 'overlay': overlay}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
