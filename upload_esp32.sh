#!/bin/bash
# ESP32 CUDA VM 上傳腳本
# 用於編譯並上傳 esp32_cuda_vm 到 ESP32 開發板

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 預設配置
# PORT="/dev/cu.usbserial-589A0095521"
PORT="/dev/cu.usbserial-2130"
FQBN="esp32:esp32:esp32"
SKETCH_PATH="examples/esp32_cuda_vm/esp32_cuda_vm.ino"

# 顯示標題
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   ESP32 CUDA VM 上傳工具${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 允許從命令列參數覆蓋 PORT
if [ ! -z "$1" ]; then
    PORT="$1"
    echo -e "${YELLOW}使用自定義序列埠: ${PORT}${NC}"
fi

# 顯示配置
echo -e "${GREEN}配置資訊:${NC}"
echo -e "  📁 Sketch: ${SKETCH_PATH}"
echo -e "  🔌 Port:   ${PORT}"
echo -e "  🎯 FQBN:   ${FQBN}"
echo ""

# 檢查 Python 腳本是否存在
if [ ! -f "cli.py" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 cli.py${NC}"
    echo -e "${YELLOW}請確保在專案根目錄執行此腳本${NC}"
    exit 1
fi

# 檢查 Sketch 是否存在
if [ ! -f "$SKETCH_PATH" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 ${SKETCH_PATH}${NC}"
    exit 1
fi

# 執行上傳
echo -e "${BLUE}🚀 開始上傳...${NC}"
echo ""

python3 cli.py upload "$SKETCH_PATH" -p "$PORT" --fqbn "$FQBN" -v

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ 上傳成功！${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}💡 提示: 使用以下指令監測序列埠輸出:${NC}"
    echo -e "   ${BLUE}python3 cli.py monitor -p ${PORT} -b 115200${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ 上傳失敗${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi