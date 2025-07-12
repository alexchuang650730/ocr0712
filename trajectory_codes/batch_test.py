#!/usr/bin/env python3
"""
OCR軌跡模擬批量測試腳本
"""

import os
import sys
import importlib.util
import time

def test_generated_codes():
    """測試所有生成的識別代碼"""
    
    print("🔄 開始批量測試生成的OCR代碼...")
    
    results = []
    test_files = [f for f in os.listdir('.') if f.startswith('ocr_recognize_') and f.endswith('.py')]
    
    for i, filename in enumerate(test_files):
        print(f"\n測試 {i+1}/{len(test_files)}: {filename}")
        
        try:
            # 動態導入模塊
            spec = importlib.util.spec_from_file_location("test_module", filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 執行主函數
            start_time = time.time()
            predicted_char, confidence = module.main()
            execution_time = time.time() - start_time
            
            # 獲取目標字符
            target_char = module.TARGET_CHARACTER
            
            # 記錄結果
            result = {
                'filename': filename,
                'target_char': target_char,
                'predicted_char': predicted_char,
                'confidence': confidence,
                'execution_time': execution_time,
                'success': predicted_char == target_char
            }
            results.append(result)
            
            status = "✅" if result['success'] else "❌"
            print(f"{status} 結果: {predicted_char} (置信度: {confidence:.3f}, 時間: {execution_time:.3f}s)")
            
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            results.append({
                'filename': filename,
                'error': str(e),
                'success': False
            })
    
    # 統計結果
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"\n📊 === 批量測試結果 ===")
    print(f"總測試數: {total}")
    print(f"成功數: {successful}")
    print(f"成功率: {successful/total*100:.1f}%")
    
    if successful > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success', False)) / successful
        avg_time = sum(r.get('execution_time', 0) for r in results if r.get('success', False)) / successful
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均執行時間: {avg_time:.3f}s")
    
    return results

if __name__ == "__main__":
    test_generated_codes()
