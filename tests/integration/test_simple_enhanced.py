#!/usr/bin/env python3
"""简化的增强检测器测试"""
import sys
from pathlib import Path
from src.core.pdf_processor.enhanced_detector import EnhancedFigureTableDetector
from src.core.pdf_processor.renderer import PDFRenderer
import fitz

def test_simple_enhanced():
    """测试基本的增强检测功能"""
    print("=== 简化的增强检测器测试 ===")
    
    # PDF文件路径
    pdf_path = Path("/mnt/d/Buffer/Work_B/helpother/medtuning-master/medtuning-master/data/PMC1301025.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF文件不存在: {pdf_path}")
        return
    
    print(f"📄 测试PDF: {pdf_path.name}")
    
    # 使用PyMuPDF直接检测
    doc = fitz.open(str(pdf_path))
    
    total_images = 0
    total_drawings = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 检测嵌入的图片
        images = page.get_images()
        if images:
            print(f"\n页面 {page_num + 1}: 发现 {len(images)} 个嵌入图片")
            total_images += len(images)
            
            # 获取图片位置
            for img_idx, img in enumerate(images):
                xref = img[0]
                rects = page.get_image_rects(xref)
                for rect in rects:
                    print(f"  图片 {img_idx}: 位置 ({int(rect.x0)}, {int(rect.y0)}) - ({int(rect.x1)}, {int(rect.y1)})")
        
        # 检测绘图元素
        drawings = page.get_drawings()
        if drawings:
            print(f"页面 {page_num + 1}: 发现 {len(drawings)} 个绘图元素")
            total_drawings += len(drawings)
    
    page_count = len(doc)
    doc.close()
    
    print(f"\n📊 总结:")
    print(f"- 总页数: {page_count}")
    print(f"- 嵌入图片总数: {total_images}")
    print(f"- 绘图元素总数: {total_drawings}")
    
    # 使用渲染器保存一些页面
    output_dir = Path("output/simple_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存前3页到: {output_dir}")
    renderer = PDFRenderer(pdf_path)
    
    for i in range(min(3, renderer.page_count)):
        output_path = output_dir / f"page_{i}.png"
        img = renderer.render_page(i, output_path)
        print(f"  ✅ 保存页面 {i}: {output_path.name}")
    
    renderer.close()
    
    print("\n✨ 测试完成!")

if __name__ == "__main__":
    test_simple_enhanced()