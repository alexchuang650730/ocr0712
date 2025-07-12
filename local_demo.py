#!/usr/bin/env python3
"""
OCR0712 本地演示系統
支持多種OCR模式的交互式演示
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

# 導入OCR系統
try:
    from sota_ondevice_ocr import SOTAOnDeviceOCR, OCRTester
    from hybrid_edge_cloud_ocr import HybridEdgeCloudOCR
    from ocr_demo import SimplifiedOCRDemo
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保所有OCR模塊都在同一目錄下")
    sys.exit(1)

class LocalOCRDemo:
    """本地OCR演示系統"""
    
    def __init__(self):
        print("🚀 初始化 OCR0712 本地演示系統...")
        
        # 初始化各種OCR系統
        self.sota_ocr = SOTAOnDeviceOCR()
        self.hybrid_ocr = HybridEdgeCloudOCR()
        self.simple_ocr = SimplifiedOCRDemo()
        
        print("✅ 所有OCR系統初始化完成")
        
    def create_demo_images(self):
        """創建演示圖像"""
        
        print("\n📷 創建演示圖像...")
        
        demo_images = {}
        
        # 1. 簡單文本圖像
        img_simple = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img_simple, "Hello OCR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(img_simple, "你好 OCR", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("demo_simple.jpg", img_simple)
        demo_images["simple"] = "demo_simple.jpg"
        
        # 2. 表格圖像
        img_table = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # 畫表格邊框
        cv2.rectangle(img_table, (50, 50), (450, 250), (0, 0, 0), 2)
        cv2.line(img_table, (250, 50), (250, 250), (0, 0, 0), 1)
        cv2.line(img_table, (50, 150), (450, 150), (0, 0, 0), 1)
        # 添加文字
        cv2.putText(img_table, "Name", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img_table, "Age", (300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img_table, "Alice", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img_table, "25", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite("demo_table.jpg", img_table)
        demo_images["table"] = "demo_table.jpg"
        
        # 3. 手寫風格圖像
        img_handwriting = np.ones((250, 400, 3), dtype=np.uint8) * 255
        # 模擬手寫線條
        points = np.array([[50, 100], [100, 120], [150, 110], [200, 130], [250, 125]], np.int32)
        cv2.polylines(img_handwriting, [points], False, (0, 0, 0), 3)
        points2 = np.array([[50, 150], [120, 160], [180, 155], [250, 165]], np.int32)
        cv2.polylines(img_handwriting, [points2], False, (0, 0, 0), 3)
        cv2.imwrite("demo_handwriting.jpg", img_handwriting)
        demo_images["handwriting"] = "demo_handwriting.jpg"
        
        # 4. 中文圖像
        img_chinese = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img_chinese, "繁體中文測試", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_chinese, "简体中文测试", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_chinese, "Mixed 混合 Text", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite("demo_chinese.jpg", img_chinese)
        demo_images["chinese"] = "demo_chinese.jpg"
        
        print(f"✅ 創建了 {len(demo_images)} 個演示圖像")
        return demo_images
        
    def interactive_demo(self):
        """交互式演示"""
        
        print("\n🎮 進入交互式演示模式")
        print("="*60)
        
        # 創建演示圖像
        demo_images = self.create_demo_images()
        
        while True:
            print("\n📋 可用的演示選項:")
            print("1. SOTA設備端OCR演示")
            print("2. 混合雲端OCR演示") 
            print("3. 簡化OCR演示")
            print("4. 比較所有OCR系統")
            print("5. 自定義圖像測試")
            print("6. 性能基準測試")
            print("0. 退出")
            
            choice = input("\n請選擇演示模式 (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.demo_sota_ocr(demo_images)
            elif choice == "2":
                self.demo_hybrid_ocr(demo_images)
            elif choice == "3":
                self.demo_simple_ocr(demo_images)
            elif choice == "4":
                self.compare_all_systems(demo_images)
            elif choice == "5":
                self.custom_image_test()
            elif choice == "6":
                self.benchmark_test(demo_images)
            else:
                print("❌ 無效選擇，請重新輸入")
                
        print("\n👋 感謝使用OCR0712演示系統！")
        
    def demo_sota_ocr(self, demo_images):
        """演示SOTA OCR系統"""
        
        print("\n🔬 SOTA設備端OCR演示")
        print("-" * 40)
        
        print("\n可用的測試圖像:")
        for i, (name, path) in enumerate(demo_images.items(), 1):
            print(f"{i}. {name} ({path})")
            
        try:
            choice = int(input("\n選擇圖像 (1-4): ")) - 1
            image_name = list(demo_images.keys())[choice]
            image_path = demo_images[image_name]
            
            print(f"\n🔍 使用SOTA OCR處理: {image_path}")
            
            start_time = time.time()
            result = self.sota_ocr.recognize(image_path)
            end_time = time.time()
            
            print(f"\n📊 SOTA OCR結果:")
            print(f"識別文本: {result.text}")
            print(f"文字類型: {result.script_type.value}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"處理時間: {end_time - start_time:.3f}秒")
            
            if result.trajectory_code:
                print(f"\n🎯 軌跡代碼生成: 已生成")
                show_code = input("是否顯示軌跡代碼? (y/n): ").lower() == 'y'
                if show_code:
                    print("\n軌跡代碼:")
                    print(result.trajectory_code[:500] + "..." if len(result.trajectory_code) > 500 else result.trajectory_code)
                    
        except (ValueError, IndexError):
            print("❌ 無效選擇")
            
    def demo_hybrid_ocr(self, demo_images):
        """演示混合雲端OCR系統"""
        
        print("\n☁️ 混合雲端OCR演示")
        print("-" * 40)
        
        print("\n可用的測試圖像:")
        for i, (name, path) in enumerate(demo_images.items(), 1):
            print(f"{i}. {name} ({path})")
            
        try:
            choice = int(input("\n選擇圖像 (1-4): ")) - 1
            image_name = list(demo_images.keys())[choice]
            image_path = demo_images[image_name]
            
            print(f"\n🔍 使用混合OCR處理: {image_path}")
            
            start_time = time.time()
            result = self.hybrid_ocr.recognize(image_path)
            end_time = time.time()
            
            print(f"\n📊 混合OCR結果:")
            print(f"識別文本: {result.get('text', 'N/A')}")
            print(f"總體置信度: {result.get('overall_confidence', 0):.3f}")
            print(f"處理時間: {end_time - start_time:.3f}秒")
            print(f"使用策略: {result.get('strategy', 'N/A')}")
            
            if result.get('edge_result'):
                print(f"\n🏠 邊緣結果: {result['edge_result'].get('text', 'N/A')}")
                print(f"邊緣置信度: {result['edge_result'].get('confidence', 0):.3f}")
                
            if result.get('cloud_result'):
                print(f"\n☁️ 雲端結果: {result['cloud_result'].get('text', 'N/A')}")
                print(f"雲端置信度: {result['cloud_result'].get('confidence', 0):.3f}")
                
        except (ValueError, IndexError):
            print("❌ 無效選擇")
            
    def demo_simple_ocr(self, demo_images):
        """演示簡化OCR系統"""
        
        print("\n🎯 簡化OCR演示")
        print("-" * 40)
        
        print("\n運行簡化OCR演示...")
        self.simple_ocr.run_interactive_demo()
        
    def compare_all_systems(self, demo_images):
        """比較所有OCR系統"""
        
        print("\n⚖️ OCR系統比較測試")
        print("=" * 60)
        
        # 選擇測試圖像
        print("\n可用的測試圖像:")
        for i, (name, path) in enumerate(demo_images.items(), 1):
            print(f"{i}. {name} ({path})")
            
        try:
            choice = int(input("\n選擇圖像 (1-4): ")) - 1
            image_name = list(demo_images.keys())[choice]
            image_path = demo_images[image_name]
            
            print(f"\n🔍 比較測試圖像: {image_path}")
            print("=" * 60)
            
            results = {}
            
            # 1. SOTA OCR
            print("\n1️⃣ SOTA設備端OCR:")
            start_time = time.time()
            try:
                sota_result = self.sota_ocr.recognize(image_path)
                sota_time = time.time() - start_time
                results['SOTA'] = {
                    'text': sota_result.text,
                    'confidence': sota_result.confidence,
                    'time': sota_time,
                    'success': True
                }
                print(f"   文本: {sota_result.text}")
                print(f"   置信度: {sota_result.confidence:.3f}")
                print(f"   時間: {sota_time:.3f}秒")
            except Exception as e:
                results['SOTA'] = {'success': False, 'error': str(e)}
                print(f"   ❌ 錯誤: {e}")
                
            # 2. 混合OCR
            print("\n2️⃣ 混合雲端OCR:")
            start_time = time.time()
            try:
                hybrid_result = self.hybrid_ocr.recognize(image_path)
                hybrid_time = time.time() - start_time
                results['Hybrid'] = {
                    'text': hybrid_result.get('text', ''),
                    'confidence': hybrid_result.get('overall_confidence', 0),
                    'time': hybrid_time,
                    'success': True
                }
                print(f"   文本: {hybrid_result.get('text', 'N/A')}")
                print(f"   置信度: {hybrid_result.get('overall_confidence', 0):.3f}")
                print(f"   時間: {hybrid_time:.3f}秒")
            except Exception as e:
                results['Hybrid'] = {'success': False, 'error': str(e)}
                print(f"   ❌ 錯誤: {e}")
                
            # 3. 簡化OCR
            print("\n3️⃣ 簡化OCR:")
            start_time = time.time()
            try:
                simple_result = self.simple_ocr.mock_recognize(image_path)
                simple_time = time.time() - start_time
                results['Simple'] = {
                    'text': simple_result.get('text', ''),
                    'confidence': simple_result.get('confidence', 0),
                    'time': simple_time,
                    'success': True
                }
                print(f"   文本: {simple_result.get('text', 'N/A')}")
                print(f"   置信度: {simple_result.get('confidence', 0):.3f}")
                print(f"   時間: {simple_time:.3f}秒")
            except Exception as e:
                results['Simple'] = {'success': False, 'error': str(e)}
                print(f"   ❌ 錯誤: {e}")
                
            # 比較總結
            print("\n📊 比較總結:")
            print("=" * 60)
            successful_results = {k: v for k, v in results.items() if v.get('success')}
            
            if successful_results:
                # 最高置信度
                best_confidence = max(successful_results.items(), key=lambda x: x[1]['confidence'])
                print(f"🎯 最高置信度: {best_confidence[0]} ({best_confidence[1]['confidence']:.3f})")
                
                # 最快速度
                fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
                print(f"⚡ 最快處理: {fastest[0]} ({fastest[1]['time']:.3f}秒)")
                
                # 推薦使用
                print("\n💡 使用建議:")
                if best_confidence[1]['confidence'] > 0.9:
                    print(f"   推薦使用 {best_confidence[0]} (高置信度)")
                elif fastest[1]['time'] < 1.0:
                    print(f"   推薦使用 {fastest[0]} (快速處理)")
                else:
                    print("   根據具體需求選擇合適的OCR系統")
            
        except (ValueError, IndexError):
            print("❌ 無效選擇")
            
    def custom_image_test(self):
        """自定義圖像測試"""
        
        print("\n📁 自定義圖像測試")
        print("-" * 40)
        
        image_path = input("請輸入圖像路徑 (相對或絕對路徑): ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ 文件不存在: {image_path}")
            return
            
        print("\n選擇OCR系統:")
        print("1. SOTA設備端OCR")
        print("2. 混合雲端OCR")
        print("3. 簡化OCR")
        
        try:
            choice = int(input("選擇 (1-3): "))
            
            print(f"\n🔍 處理圖像: {image_path}")
            
            if choice == 1:
                result = self.sota_ocr.recognize(image_path)
                print(f"\n📊 SOTA OCR結果:")
                print(f"識別文本: {result.text}")
                print(f"置信度: {result.confidence:.3f}")
            elif choice == 2:
                result = self.hybrid_ocr.recognize(image_path)
                print(f"\n📊 混合OCR結果:")
                print(f"識別文本: {result.get('text', 'N/A')}")
                print(f"置信度: {result.get('overall_confidence', 0):.3f}")
            elif choice == 3:
                result = self.simple_ocr.mock_recognize(image_path)
                print(f"\n📊 簡化OCR結果:")
                print(f"識別文本: {result.get('text', 'N/A')}")
                print(f"置信度: {result.get('confidence', 0):.3f}")
            else:
                print("❌ 無效選擇")
                
        except ValueError:
            print("❌ 無效輸入")
        except Exception as e:
            print(f"❌ 處理錯誤: {e}")
            
    def benchmark_test(self, demo_images):
        """性能基準測試"""
        
        print("\n🏁 性能基準測試")
        print("=" * 60)
        
        test_rounds = 3
        print(f"📈 將對每個系統進行 {test_rounds} 輪測試...")
        
        benchmark_results = {}
        
        for system_name, ocr_system in [
            ("SOTA OCR", self.sota_ocr),
            ("Hybrid OCR", self.hybrid_ocr),
            ("Simple OCR", self.simple_ocr)
        ]:
            print(f"\n🔄 測試 {system_name}...")
            
            times = []
            successes = 0
            
            for round_num in range(test_rounds):
                print(f"  輪次 {round_num + 1}/{test_rounds}")
                
                round_times = []
                
                for img_name, img_path in demo_images.items():
                    try:
                        start_time = time.time()
                        
                        if system_name == "SOTA OCR":
                            result = ocr_system.recognize(img_path)
                        elif system_name == "Hybrid OCR":
                            result = ocr_system.recognize(img_path)
                        else:  # Simple OCR
                            result = ocr_system.mock_recognize(img_path)
                            
                        end_time = time.time()
                        round_times.append(end_time - start_time)
                        successes += 1
                        
                    except Exception as e:
                        print(f"    ❌ {img_name} 處理失敗: {e}")
                        
                if round_times:
                    times.extend(round_times)
                    
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                success_rate = successes / (test_rounds * len(demo_images))
                
                benchmark_results[system_name] = {
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'success_rate': success_rate
                }
                
                print(f"  ✅ 平均時間: {avg_time:.3f}秒")
                print(f"  ⚡ 最快時間: {min_time:.3f}秒")
                print(f"  🐌 最慢時間: {max_time:.3f}秒")
                print(f"  📊 成功率: {success_rate:.1%}")
            else:
                print(f"  ❌ {system_name} 測試失敗")
                
        # 基準測試總結
        print("\n📊 基準測試總結:")
        print("=" * 60)
        
        if benchmark_results:
            # 最快系統
            fastest_system = min(benchmark_results.items(), key=lambda x: x[1]['avg_time'])
            print(f"🥇 最快系統: {fastest_system[0]} (平均 {fastest_system[1]['avg_time']:.3f}秒)")
            
            # 最穩定系統
            most_stable = min(benchmark_results.items(), 
                            key=lambda x: x[1]['max_time'] - x[1]['min_time'])
            print(f"🎯 最穩定系統: {most_stable[0]} (時間變化 {most_stable[1]['max_time'] - most_stable[1]['min_time']:.3f}秒)")
            
            # 最可靠系統
            most_reliable = max(benchmark_results.items(), key=lambda x: x[1]['success_rate'])
            print(f"🛡️ 最可靠系統: {most_reliable[0]} (成功率 {most_reliable[1]['success_rate']:.1%})")
            
    def batch_process(self, input_dir):
        """批量處理目錄中的圖像"""
        
        print(f"\n📁 批量處理目錄: {input_dir}")
        
        if not os.path.exists(input_dir):
            print(f"❌ 目錄不存在: {input_dir}")
            return
            
        # 支持的圖像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 找到所有圖像文件
        image_files = []
        for file_path in Path(input_dir).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
                
        if not image_files:
            print(f"❌ 在目錄中未找到圖像文件: {input_dir}")
            return
            
        print(f"📷 找到 {len(image_files)} 個圖像文件")
        
        # 選擇OCR系統
        print("\n選擇OCR系統:")
        print("1. SOTA設備端OCR")
        print("2. 混合雲端OCR")
        print("3. 簡化OCR")
        
        try:
            choice = int(input("選擇 (1-3): "))
            
            if choice == 1:
                ocr_system = self.sota_ocr
                process_func = lambda path: self.sota_ocr.recognize(path)
            elif choice == 2:
                ocr_system = self.hybrid_ocr
                process_func = lambda path: self.hybrid_ocr.recognize(path)
            elif choice == 3:
                ocr_system = self.simple_ocr
                process_func = lambda path: self.simple_ocr.mock_recognize(path)
            else:
                print("❌ 無效選擇")
                return
                
        except ValueError:
            print("❌ 無效輸入")
            return
            
        # 批量處理
        results = []
        success_count = 0
        
        print(f"\n🔄 開始批量處理...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📷 處理 {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                start_time = time.time()
                result = process_func(image_path)
                end_time = time.time()
                
                if choice in [1]:  # SOTA OCR
                    text = result.text
                    confidence = result.confidence
                elif choice == 2:  # Hybrid OCR
                    text = result.get('text', '')
                    confidence = result.get('overall_confidence', 0)
                else:  # Simple OCR
                    text = result.get('text', '')
                    confidence = result.get('confidence', 0)
                    
                results.append({
                    'file': image_path,
                    'text': text,
                    'confidence': confidence,
                    'time': end_time - start_time,
                    'success': True
                })
                
                success_count += 1
                print(f"   ✅ 成功: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"   置信度: {confidence:.3f}, 時間: {end_time - start_time:.3f}秒")
                
            except Exception as e:
                results.append({
                    'file': image_path,
                    'error': str(e),
                    'success': False
                })
                print(f"   ❌ 失敗: {e}")
                
        # 保存結果
        output_file = f"batch_results_{int(time.time())}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR批量處理結果\n")
            f.write(f"處理時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總文件數: {len(image_files)}\n")
            f.write(f"成功數: {success_count}\n")
            f.write(f"失敗數: {len(image_files) - success_count}\n")
            f.write(f"成功率: {success_count/len(image_files):.1%}\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"文件 {i}: {result['file']}\n")
                if result['success']:
                    f.write(f"文本: {result['text']}\n")
                    f.write(f"置信度: {result['confidence']:.3f}\n")
                    f.write(f"處理時間: {result['time']:.3f}秒\n")
                else:
                    f.write(f"錯誤: {result['error']}\n")
                f.write("-" * 40 + "\n")
                
        print(f"\n📄 批量處理完成！結果已保存到: {output_file}")
        print(f"📊 總計: {len(image_files)} 個文件, 成功: {success_count}, 失敗: {len(image_files) - success_count}")
        print(f"🎯 成功率: {success_count/len(image_files):.1%}")

def main():
    """主函數"""
    
    parser = argparse.ArgumentParser(description="OCR0712 本地演示系統")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式演示模式")
    parser.add_argument("--batch", "-b", action="store_true", help="批量處理模式")
    parser.add_argument("--input_dir", "-d", default="./test_images", help="批量處理的輸入目錄")
    parser.add_argument("--benchmark", action="store_true", help="性能基準測試")
    
    args = parser.parse_args()
    
    # 歡迎信息
    print("🚀 歡迎使用 OCR0712 本地演示系統")
    print("=" * 60)
    print("基於中文OCR的先進識別系統")
    print("支持 SOTA設備端OCR、混合雲端OCR 和簡化OCR")
    print("=" * 60)
    
    try:
        demo = LocalOCRDemo()
        
        if args.interactive:
            demo.interactive_demo()
        elif args.batch:
            demo.batch_process(args.input_dir)
        elif args.benchmark:
            demo_images = demo.create_demo_images()
            demo.benchmark_test(demo_images)
        else:
            # 默認運行交互式演示
            print("\n💡 使用 --help 查看所有選項")
            print("🎮 啟動交互式演示...")
            demo.interactive_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 用戶中斷，退出演示系統")
    except Exception as e:
        print(f"\n❌ 系統錯誤: {e}")
        
if __name__ == "__main__":
    main()