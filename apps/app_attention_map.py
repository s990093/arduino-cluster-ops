"""
ESP32 Transformer Self-Attention Map Visualization
Generates and visualizes attention heatmaps from a simple Vision Transformer
Image size: 64x64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import struct
import serial
import time

# Configuration
IMG_PATH = "/Users/hungwei/Desktop/Proj/arduino-cluster-ops/IMG_6257.JPG"
IMG_SIZE = 64
PATCH_SIZE = 8  # 64/8 = 8x8 = 64 patches
EMBED_DIM = 128
NUM_HEADS = 4
PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800

class SimpleViT(nn.Module):
    """Minimal Vision Transformer for attention visualization"""
    def __init__(self, img_size=64, patch_size=8, embed_dim=128, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Track attention weights
        self.attention_weights = None
        
    def forward(self, x):
        # x: (B, 3, 64, 64)
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights  # (B, num_heads, 64, 64)
        
        return attn_output

def load_and_preprocess_image(img_path, size=64):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img, img_tensor

def visualize_attention_maps(attention_weights, original_img, save_path="attention_maps.png"):
    """
    Visualize multi-head attention maps
    attention_weights: (1, num_heads, num_patches, num_patches)
    """
    num_heads = attention_weights.shape[1]
    num_patches = int(np.sqrt(attention_weights.shape[2]))
    
    # Create figure
    fig, axes = plt.subplots(2, num_heads + 1, figsize=(18, 8))
    
    # Show original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image (64x64)", fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # For each attention head
    for i in range(num_heads):
        # Get attention for this head, averaged over all query positions
        attn_head = attention_weights[0, i].cpu().numpy()  # (64, 64)
        
        # Average attention map (mean over queries)
        avg_attn = attn_head.mean(axis=0).reshape(num_patches, num_patches)
        
        # Show average attention map
        im = axes[0, i+1].imshow(avg_attn, cmap='hot', interpolation='nearest')
        axes[0, i+1].set_title(f"Head {i+1} Avg Attention", fontsize=10)
        axes[0, i+1].axis('off')
        plt.colorbar(im, ax=axes[0, i+1], fraction=0.046)
        
        # Show specific query position attention (e.g., center patch)
        center_idx = num_patches * num_patches // 2
        query_attn = attn_head[center_idx].reshape(num_patches, num_patches)
        
        im2 = axes[1, i+1].imshow(query_attn, cmap='viridis', interpolation='nearest')
        axes[1, i+1].set_title(f"Head {i+1} Center Query", fontsize=10)
        axes[1, i+1].axis('off')
        plt.colorbar(im2, ax=axes[1, i+1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Attention maps saved to: {save_path}")
    plt.show()

def main():
    print("🔍 Transformer Self-Attention Map Visualization")
    print("=" * 60)
    
    # Load image
    print(f"📷 Loading image: {IMG_PATH}")
    original_img, img_tensor = load_and_preprocess_image(IMG_PATH, IMG_SIZE)
    print(f"   Image size: {img_tensor.shape}")
    
    # Create model
    print(f"🧠 Creating ViT model...")
    print(f"   Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"   Number of patches: {(IMG_SIZE // PATCH_SIZE) ** 2}")
    print(f"   Embedding dim: {EMBED_DIM}")
    print(f"   Attention heads: {NUM_HEADS}")
    
    model = SimpleViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )
    model.eval()
    
    # Forward pass to get attention
    print("🔥 Running forward pass...")
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Get attention weights
    attention_weights = model.attention_weights
    print(f"   Attention shape: {attention_weights.shape}")
    print(f"   Range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    
    # Visualize
    print("🎨 Generating attention visualizations...")
    visualize_attention_maps(attention_weights, original_img)
    
    # Print attention statistics
    print("\n📊 Attention Statistics:")
    for i in range(NUM_HEADS):
        attn = attention_weights[0, i].cpu().numpy()
        print(f"   Head {i+1}: Mean={attn.mean():.4f}, Std={attn.std():.4f}, "
              f"Max={attn.max():.4f}")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()
