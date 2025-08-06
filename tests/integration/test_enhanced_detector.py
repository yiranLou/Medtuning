#!/usr/bin/env python3
"""测试增强的图表和表格检测器"""
import sys
from pathlib import Path
from src.core.pdf_processor.enhanced_detector import EnhancedFigureTableDetector
from src.core.pdf_processor.renderer import PDFRenderer

def test_enhanced_detector():
    """测试增强检测器"""
    print("=== 测试增强的图表和表格检测器 ===")
    
    # PDF文件路径
    pdf_paths = [
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/PMC1301025.pdf"),
        Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/2309.09431v4.pdf")
    ]
    
    output_dir = Path("output/enhanced_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = EnhancedFigureTableDetector()
    
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"⚠️ PDF文件不存在: {pdf_path}")
            continue
        
        print(f"\n📄 处理PDF: {pdf_path.name}")
        
        # 检测所有元素
        results = detector.detect_all_elements(pdf_path)
        
        print(f"\n检测结果:")
        print(f"- 图表: {len(results['figures'])}个")
        print(f"- 表格: {len(results['tables'])}个")
        print(f"- 公式: {len(results['equations'])}个")
        
        # 渲染检测到的元素
        renderer = PDFRenderer(pdf_path)
        
        # 保存图表
        for i, fig in enumerate(results['figures']):
            output_path = output_dir / f"{pdf_path.stem}_figure_{i}.png"
            try:
                img = renderer.crop_region(fig.page_index, fig.bbox, output_path)
                print(f"  ✅ 保存图表 {i}: {output_path.name}")
                if fig.caption:
                    print(f"     标题: {fig.caption[:50]}...")
            except Exception as e:
                print(f"  ❌ 保存图表 {i} 失败: {e}")
        
        # 保存表格
        for i, table in enumerate(results['tables']):
            output_path = output_dir / f"{pdf_path.stem}_table_{i}.png"
            try:
                img = renderer.crop_region(table.page_index, table.bbox, output_path)
                print(f"  ✅ 保存表格 {i}: {output_path.name}")
                if table.caption:
                    print(f"     标题: {table.caption[:50]}...")
            except Exception as e:
                print(f"  ❌ 保存表格 {i} 失败: {e}")
        
        # 保存公式
        for i, eq in enumerate(results['equations'][:5]):  # 只保存前5个公式
            output_path = output_dir / f"{pdf_path.stem}_equation_{i}.png"
            try:
                img = renderer.crop_region(eq.page_index, eq.bbox, output_path)
                print(f"  ✅ 保存公式 {i}: {output_path.name}")
                if eq.caption:
                    print(f"     内容: {eq.caption[:50]}...")
            except Exception as e:
                print(f"  ❌ 保存公式 {i} 失败: {e}")
        
        renderer.close()
    
    print(f"\n✨ 所有结果保存在: {output_dir}")

if __name__ == "__main__":
    test_enhanced_detector()