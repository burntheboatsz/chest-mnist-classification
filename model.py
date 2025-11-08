# model.py

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0Binary(nn.Module):
    """
    EfficientNet-B0 pre-trained model untuk klasifikasi biner.
    EfficientNet sangat efisien dan cocok untuk small images.
    Menggunakan transfer learning dari ImageNet dengan adaptasi untuk:
    - Input grayscale (1 channel)
    - Output binary classification (1 class dengan sigmoid)
    """
    
    def __init__(
        self,
        img_size=28,
        in_channels=1,
        num_classes=1,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=False
    ):
        super(EfficientNetB0Binary, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet-B0 dari torchvision
        print(f"\nLoading pre-trained EfficientNet-B0 from ImageNet...")
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(weights=None)
        
        # Extract features (convolutional backbone)
        self.features = efficientnet.features
        
        # Modifikasi first conv untuk input grayscale (1 channel -> 3 channel)
        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        if pretrained:
            # Copy dan rata-rata weight dari RGB ke grayscale
            with torch.no_grad():
                self.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Freeze backbone if requested (for progressive training)
        if freeze_backbone:
            print("⚠️  Freezing backbone layers...")
            for param in self.features.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen (only classifier trainable)")
        else:
            print("✓ All layers trainable from start")
        
        # Global pooling (adaptive untuk handle berbagai ukuran input)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Modifikasi classifier untuk binary classification
        # EfficientNet-B0 output feature: 1280
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 3),
            nn.Linear(256, num_classes)  # num_classes should be 1 for binary
        )
        
        print(f"Model loaded successfully!")
        print(f"Architecture: EfficientNet-B0 (pre-trained on ImageNet)")
        print(f"Input: Grayscale ({in_channels} channel)")
        print(f"Output: Binary classification")
        print(f"Parameters: ~5M (very efficient!)")
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 1, H, W)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Extract features
        features = self.features(x)
        
        # Apply ReLU and adaptive pooling
        features = torch.nn.functional.relu(features, inplace=True)
        features = self.adaptive_pool(features)
        
        # Flatten
        features = torch.flatten(features, 1)
        
        # Classify
        output = self.classifier(features)
        
        return output


# Alias untuk backward compatibility
DenseNet121Binary = EfficientNetB0Binary
ResNet50Binary = EfficientNetB0Binary
SimpleCNN = EfficientNetB0Binary
VisionTransformerSmall = EfficientNetB0Binary

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 1
    IN_CHANNELS = 1
    
    print("=" * 60)
    print("   --- Menguji Model 'EfficientNet-B0 (Pre-trained)' ---")
    print("=" * 60)
    
    # Setup device (GPU jika tersedia)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    print("Loading pre-trained EfficientNet-B0...")
    model = EfficientNetB0Binary(
        pretrained=True,
        num_classes=1,
        dropout=0.3
    ).to(device)
    
    print("\nModel loaded successfully!")
    print(f"Architecture: EfficientNet-B0 (pre-trained on ImageNet)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    dummy_input = torch.randn(32, IN_CHANNELS, 64, 64).to(device)
    print(f"\n🧪 Testing forward pass with 64x64 input...")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Device input: {dummy_input.device}")
    print(f"Device output: {output.device}")
    
    print("\n" + "=" * 60)
    print("✅ Pengujian model 'EfficientNet-B0 (Pre-trained)' berhasil!")
    print("=" * 60)
