"""
ESP32 Transformer Attention - Python Client
Computes self-attention on ESP32 for 64x64 image (64 patches)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import serial
import struct
import time
from pathlib import Path

# Configuration
IMG_PATH = "/Users/hungwei/Desktop/Proj/arduino-cluster-ops/IMG_6257.JPG"
PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800
IMG_SIZE = 64
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64
EMBED_DIM = 32  # Simplified for ESP32

# VRAM addresses
VRAM_Q = 0x0000      # Q matrix: 64x32 floats (~8KB)
VRAM_K = 0x2000      # K matrix: 64x32 floats
VRAM_QK = 0x4000     # QK result: 64x64 floats (~16KB)
VRAM_V = 0x8000      # V matrix
VRAM_OUTPUT = 0xC000  # Output

class ESP32AttentionClient:
    """Client for ESP32 attention computation - based on ESP32TurboClient"""
    def __init__(self, port, baud=460800):
        self.ser = serial.Serial(port, baud, timeout=2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(2)  # Wait for ESP32 reset
        
        # Clear any startup messages
        while self.ser.in_waiting:
            self.ser.readline()
        
        print(f"✅ Connected to ESP32 at {baud}")
    
    def load_kernel(self, kernel_path):
        """Load kernel binary to ESP32 IMEM"""
        with open(kernel_path, 'rb') as f:
            kernel_bin = f.read()
        
        total_size = len(kernel_bin)
        print(f"📤 Uploading kernel: {total_size} bytes")
        
        # Send load command
        self.ser.write(f"load_imem {total_size}\n".encode())
        
        # Wait for ACK
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"   [ESP32] {line}")
            if "ACK" in line:
                break
        
        # Send all kernel data at once (like app_edge_conv.py)
        print(f"   Sending binary ({len(kernel_bin)} bytes)...")
        self.ser.write(kernel_bin)
        
        # Wait for completion
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"   [ESP32] {line}")
            if "OK" in line or "LOADED" in line:
                break
        
        print(f"   ✅ Kernel uploaded")
    
    def h2d(self, addr, data_bytes):
        """Host to device memory transfer"""
        size = len(data_bytes)
        
        print(f"   📤 H2D: {size} bytes to {hex(addr)}")
        
        # Send DMA command (note: size in bytes, not words)
        self.ser.write(f"dma_h2d {hex(addr)} {size}\n".encode())
        
        # Wait for ACK
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"      [ESP32] {line}")
            if "ACK" in line:
                break
        
        # Send all data at once
        self.ser.write(data_bytes)
        
        # Wait for DMA completion
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"      [ESP32] {line}")
            if "DMA" in line or "OK" in line:
                break
        
        print(f"   ✅ Upload complete")
    
    def d2h(self, addr, size_bytes):
        """Device to host memory transfer"""
        count = size_bytes // 4
        print(f"   📥 D2H: {count} words from {hex(addr)}")
        
        self.ser.write(f"dma_d2h_binary {hex(addr)} {count}\n".encode())
        
        # Wait for header
        actual_bytes = 0
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"      [ESP32] {line}")
            if "ACK_D2H_BIN" in line:
                actual_bytes = int(line.split(":")[1])
                break
        
        # Read binary data
        data = self.ser.read(actual_bytes)
        
        # Wait for confirmation
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line and "D2H_OK" in line:
                break
        
        # Convert to int32 array
        result = np.frombuffer(data, dtype=np.int32)
        print(f"   ✅ Downloaded {len(result)} values")
        return result
    
    def launch(self):
        """Launch kernel execution"""
        print(f"   🚀 Launching kernel...")
        self.ser.write(b"kernel_launch\n")
        
        start = time.time()
        # Wait for EXIT
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"      [ESP32] {line}")
            if "EXIT" in line:
                break
        
        elapsed = time.time() - start
        print(f"   ✅ Kernel completed in {elapsed:.3f}s")

def load_image(img_path, size=64):
    """Load and preprocess image to patches"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    
    # Create patches
    patches = img_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(3, -1, PATCH_SIZE, PATCH_SIZE)  # (3, 64, 8, 8)
    patches = patches.permute(1, 0, 2, 3).reshape(NUM_PATCHES, -1)  # (64, 192)
    
    return img, patches

def create_qkv_projections(patches, embed_dim=32):
    """Create Q, K, V from patches using simple linear projection"""
    input_dim = patches.shape[1]  # 192 (3*8*8)
    
    # Random projection matrices (for demo)
    W_q = torch.randn(input_dim, embed_dim) * 0.1
    W_k = torch.randn(input_dim, embed_dim) * 0.1
    W_v = torch.randn(input_dim, embed_dim) * 0.1
    
    Q = patches @ W_q  # (64, 32)
    K = patches @ W_k
    V = patches @ W_v
    
    return Q, K, V

def compute_attention_pytorch(Q, K, V):
    """Reference PyTorch attention implementation"""
    d_k = Q.shape[-1]
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # (64, 64)
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention
    output = attn_weights @ V  # (64, 32)
    
    return output, attn_weights

def main():
    print("🔍 ESP32 Transformer Self-Attention")
    print("=" * 60)
    
    # Load image and create patches
    print(f"📷 Loading image: {IMG_PATH}")
    img, patches = load_image(IMG_PATH, IMG_SIZE)
    print(f"   Patches: {patches.shape}")
    
    # Create Q, K, V
    print("🧠 Creating Q, K, V matrices...")
    Q, K, V = create_qkv_projections(patches, EMBED_DIM)
    print(f"   Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    
    # Compute reference using PyTorch
    print("🔥 Computing PyTorch reference...")
    output_ref, attn_ref = compute_attention_pytorch(Q, K, V)
    print(f"   Output: {output_ref.shape}")
    print(f"   Attention: {attn_ref.shape}")
    print(f"   Attention range: [{attn_ref.min():.4f}, {attn_ref.max():.4f}]")
    
    # ESP32 computation
    print("\n🚀 ESP32 Computation Pipeline")
    print("-" * 60)
    
    try:
        # Connect to ESP32
        client = ESP32AttentionClient(PORT, BAUD_RATE)
        
        # Step 1: Load MatMul kernel
        print("\n📤 Step 1: Loading MatMul kernel...")
        if Path("kernel_matmul.bin").exists():
            client.load_kernel("kernel_matmul.bin")
        else:
            print("   ⚠️  kernel_matmul.bin not found, skipping ESP32 computation")
            raise FileNotFoundError("kernel_matmul.bin")
        
        # Step 2: Upload Q matrix (simplified - first 8x8 tile)
        print("\n📤 Step 2: Uploading Q matrix (8x8 tile)...")
        Q_tile = Q[:8, :8].numpy().astype(np.float32)
        Q_bytes = Q_tile.flatten().tobytes()
        client.h2d(VRAM_Q, Q_bytes)
        print(f"   Uploaded {len(Q_bytes)} bytes to {hex(VRAM_Q)}")
        
        # Step 3: Upload K^T matrix (first 8x8 tile)
        print("\n📤 Step 3: Uploading K^T matrix (8x8 tile)...")
        K_tile = K[:8, :8].T.numpy().astype(np.float32)
        K_bytes = K_tile.flatten().tobytes()
        client.h2d(VRAM_K, K_bytes)
        print(f"   Uploaded {len(K_bytes)} bytes to {hex(VRAM_K)}")
        
        # Step 4: Launch MatMul kernel (Q × K^T)
        print("\n🔥 Step 4: Launching MatMul kernel...")
        start_time = time.time()
        client.launch()
        exec_time = time.time() - start_time
        print(f"   ✅ Kernel completed in {exec_time:.3f}s")
        
        # Step 5: Download QK result
        print("\n📥 Step 5: Downloading QK result...")
        result_size = 8 * 8 * 4  # 8x8 floats
        qk_result = client.d2h(VRAM_QK, result_size)
        qk_matrix = qk_result.view(np.float32).reshape(8, 8)
        print(f"   Downloaded {len(qk_result)} values")
        print(f"   QK range: [{qk_matrix.min():.4f}, {qk_matrix.max():.4f}]")
        
        # Compare with reference (first 8x8 tile)
        ref_qk = (Q[:8, :8] @ K[:8, :8].T).numpy()
        diff = np.abs(qk_matrix - ref_qk)
        print(f"\n📊 Verification (8x8 tile):")
        print(f"   Reference QK range: [{ref_qk.min():.4f}, {ref_qk.max():.4f}]")
        print(f"   Max difference: {diff.max():.4f}")
        print(f"   Mean absolute error: {diff.mean():.4f}")
        
        if diff.max() < 0.01:
            print("   ✅ ESP32 result matches reference!")
        else:
            print("   ⚠️  Significant difference detected")
        
    except (FileNotFoundError, serial.SerialException) as e:
        print(f"\n⚠️  ESP32 computation skipped: {e}")
        print("   Using PyTorch reference only")
    
    # Visualize reference attention
    print("\n🎨 Visualizing attention maps...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image (64x64)")
    axes[0].axis('off')
    
    # Average attention (mean over all queries)
    avg_attn = attn_ref.mean(dim=0).numpy().reshape(8, 8)
    im1 = axes[1].imshow(avg_attn, cmap='hot', interpolation='nearest')
    axes[1].set_title("Average Attention Map")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Center query attention
    center_idx = NUM_PATCHES // 2
    center_attn = attn_ref[center_idx].numpy().reshape(8, 8)
    im2 = axes[2].imshow(center_attn, cmap='viridis', interpolation='nearest')
    axes[2].set_title(f"Center Patch Attention")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("attention_esp32_result.png", dpi=150)
    print("✅ Saved: attention_esp32_result.png")
    plt.show()
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()
