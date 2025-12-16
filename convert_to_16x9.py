#!/usr/bin/env python3
"""
照片轉換為 16:9 比例,使用白色填充
Convert images to 16:9 aspect ratio with white padding
"""

from PIL import Image
import os
import sys
import argparse


def convert_to_16x9(input_path, output_path=None, quality=95):
    """
    將圖片轉換為 16:9 比例,使用白色填充
    
    Args:
        input_path: 輸入圖片路徑
        output_path: 輸出圖片路徑 (如果為 None,會自動生成)
        quality: JPEG 品質 (1-100)
    """
    try:
        # 開啟圖片
        img = Image.open(input_path)
        original_width, original_height = img.size
        
        print(f"處理圖片: {input_path}")
        print(f"原始尺寸: {original_width} x {original_height}")
        
        # 目標比例 16:9
        target_ratio = 16 / 9
        current_ratio = original_width / original_height
        
        # 計算新尺寸
        if current_ratio > target_ratio:
            # 圖片太寬,需要上下填充
            new_width = original_width
            new_height = int(original_width / target_ratio)
            padding_top = (new_height - original_height) // 2
            padding_left = 0
        else:
            # 圖片太高,需要左右填充
            new_height = original_height
            new_width = int(original_height * target_ratio)
            padding_left = (new_width - original_width) // 2
            padding_top = 0
        
        # 建立白色背景
        new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        
        # 將原圖貼上到白色背景
        if img.mode == 'RGBA':
            # 如果有透明通道,先轉換為 RGB
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            new_img.paste(rgb_img, (padding_left, padding_top))
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            new_img.paste(img, (padding_left, padding_top))
        
        # 生成輸出路徑
        if output_path is None:
            base_name, ext = os.path.splitext(input_path)
            output_path = f"{base_name}_16x9{ext}"
        
        # 儲存圖片
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            new_img.save(output_path, 'JPEG', quality=quality)
        else:
            new_img.save(output_path)
        
        print(f"新尺寸: {new_width} x {new_height} (16:9)")
        print(f"已儲存至: {output_path}\n")
        
        return output_path
        
    except Exception as e:
        print(f"錯誤: 處理 {input_path} 時發生錯誤: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='將照片轉換為 16:9 比例,使用白色填充',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 轉換單張圖片
  python convert_to_16x9.py photo.jpg
  
  # 轉換多張圖片
  python convert_to_16x9.py photo1.jpg photo2.jpg photo3.png
  
  # 指定輸出檔名
  python convert_to_16x9.py input.jpg -o output.jpg
  
  # 設定 JPEG 品質
  python convert_to_16x9.py photo.jpg -q 90
  
  # 轉換資料夾內所有圖片
  python convert_to_16x9.py *.jpg
        """
    )
    
    parser.add_argument('input_files', nargs='+', help='輸入圖片檔案')
    parser.add_argument('-o', '--output', help='輸出檔案路徑 (僅適用於單一輸入檔案)')
    parser.add_argument('-q', '--quality', type=int, default=95, 
                        help='JPEG 品質 (1-100, 預設: 95)')
    
    args = parser.parse_args()
    
    # 檢查是否有多個輸入檔案時指定了輸出檔名
    if len(args.input_files) > 1 and args.output:
        print("錯誤: 處理多個檔案時不能指定單一輸出檔名")
        sys.exit(1)
    
    # 處理所有輸入檔案
    success_count = 0
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"警告: 檔案不存在: {input_file}")
            continue
        
        output_file = args.output if len(args.input_files) == 1 else None
        result = convert_to_16x9(input_file, output_file, args.quality)
        
        if result:
            success_count += 1
    
    print(f"完成! 成功轉換 {success_count}/{len(args.input_files)} 張圖片")


if __name__ == '__main__':
    main()
