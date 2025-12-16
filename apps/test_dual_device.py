#!/usr/bin/env python3
"""
Quick test script for dual-device edge convolution
Tests with a smaller image first to verify connectivity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.app_edge_conv_dual import *

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DUAL-DEVICE EDGE CONVOLUTION TEST")
    print("=" * 60)
    
    # Use smaller image for faster testing
    IMG_W, IMG_H = 256, 256
    TILE_W = 64
    TILE_H = 32
    
    print(f"\n📏 Test Configuration:")
    print(f"   Image Size: {IMG_W}x{IMG_H}")
    print(f"   Tile Size: {TILE_W}x{TILE_H}")
    print(f"   Device 0: {PORT_DEVICE_0}")
    print(f"   Device 1: {PORT_DEVICE_1}")
    
    # Test image generation (simple gradient)
    print(f"\n🎨 Generating test image...")
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for y in range(IMG_H):
        for x in range(IMG_W):
            img[y, x, 0] = x % 256  # Red channel
            img[y, x, 1] = y % 256  # Green channel
            img[y, x, 2] = (x + y) // 2 % 256  # Blue channel
    
    print(f"   Generated {img.shape} test image")
    
    # Build kernel
    PAD_W = TILE_W + 2
    PAD_H = TILE_H + 2
    print(f"\n🔨 Building kernel for {PAD_W}x{PAD_H} tiles...")
    prog_objs = build_asm_program(PAD_W, PAD_H)
    inst_vals = [int(i.to_hex(), 16) for i in prog_objs]
    kernel_bin = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    print(f"   Kernel size: {len(kernel_bin)} bytes, {len(inst_vals)} instructions")
    
    # Connect to devices
    print(f"\n🔌 Connecting to devices...")
    try:
        manager = DualDeviceManager(PORT_DEVICE_0, PORT_DEVICE_1, kernel_bin)
        print("✅ Both devices connected and ready")
        
        # Test single tile on each device
        print(f"\n🧪 Testing single tile on each device...")
        img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        
        for device_id in range(2):
            print(f"\n   Device {device_id}:")
            tile_padded = img_padded[0:PAD_H, 0:PAD_W, 0]  # First tile, red channel
            flat_tile = tile_padded.flatten().astype(np.int32)
            input_bytes = struct.pack(f'<{len(flat_tile)}I', *flat_tile)
            
            try:
                tile_out, exec_time = manager.process_task(device_id, input_bytes)
                print(f"      ✅ Success! Exec time: {exec_time*1000:.2f}ms")
                print(f"      Output shape: {tile_out.shape}")
                print(f"      Output range: [{tile_out.min()}, {tile_out.max()}]")
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        # Print device stats
        print(f"\n📊 Device Statistics:")
        stats = manager.get_stats()
        for s in stats:
            print(f"\n   Device {s['device_id']}:")
            print(f"      Tasks: {s['tasks']}")
            if 'ratio' in s:
                print(f"      Compression: {s['ratio']:.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
