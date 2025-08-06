#!/usr/bin/env python3
"""测试工作版本的增强检测器"""
import sys
from pathlib import Path
from src.core.pdf_processor.working_enhanced_detector import WorkingEnhancedDetector
from src.core.pdf_processor.renderer import PDFRenderer

def test_working_detector():
    """测试工作版本的检测器"""
    print("=== 测试工作版本的增强检测器 ===")
    
    # PDF文件路径
    pdf_paths = [
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/PMC1301025.pdf"),
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/2309.09431v4.pdf")
    ]
    
    output_dir = Path("output/working_detector_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = WorkingEnhancedDetector()
    
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"⚠️ PDF文件不存在: {pdf_path}")
            continue
        
        print(f"\n📄 处理PDF: {pdf_path.name}")
        
        # 检测所有元素
        try:
            results = detector.detect_all_elements(pdf_path)
            
            print(f"\n检测结果:")
            print(f"- 图片: {len(results['figures'])}个")
            print(f"- 表格: {len(results['tables'])}个")
            
            # 创建PDF专用目录
            pdf_output_dir = output_dir / pdf_path.stem
            pdf_output_dir.mkdir(exist_ok=True)
            
            # 使用渲染器保存检测到的图片
            renderer = PDFRenderer(pdf_path)
            
            # 保存前5个图片
            for i, fig in enumerate(results['figures'][:5]):
                output_path = pdf_output_dir / f"figure_{i}_page{fig.page_index + 1}.png"
                try:
                    img = renderer.crop_region(fig.page_index, fig.bbox, output_path)
                    print(f"  ✅ 保存图片 {i + 1}: {output_path.name}")
                    if fig.caption:
                        print(f"     标题: {fig.caption}")
                except Exception as e:
                    print(f"  ❌ 保存图片 {i + 1} 失败: {e}")
            
            # 保存前3个表格
            for i, table in enumerate(results['tables'][:3]):
                output_path = pdf_output_dir / f"table_{i}_page{table.page_index + 1}.png"
                try:
                    img = renderer.crop_region(table.page_index, table.bbox, output_path)
                    print(f"  ✅ 保存表格 {i + 1}: {output_path.name}")
                    if table.caption:
                        print(f"     标题: {table.caption}")
                except Exception as e:
                    print(f"  ❌ 保存表格 {i + 1} 失败: {e}")
            
            renderer.close()
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✨ 测试完成！结果保存在: {output_dir}")
    
    # 统计结果
    print("\n📊 统计信息:")
    for pdf_dir in output_dir.iterdir():
        if pdf_dir.is_dir():
            figures = list(pdf_dir.glob("figure_*.png"))
            tables = list(pdf_dir.glob("table_*.png"))
            print(f"- {pdf_dir.name}: {len(figures)}个图片, {len(tables)}个表格")

if __name__ == "__main__":
    test_working_detector()