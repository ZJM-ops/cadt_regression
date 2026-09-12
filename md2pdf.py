#!/usr/bin/env python3
"""
通用 Markdown 转 PDF 脚本

用法:
    python md2pdf.py input.md
    python md2pdf.py input.md -o output.pdf
    python md2pdf.py input.md --css custom.css
    python md2pdf.py input.md --backend playwright   # 默认, 支持 LaTeX 公式
    python md2pdf.py input.md --backend weasyprint    # 无浏览器离线兜底, 不渲染公式

依赖:
    playwright(默认): pip install markdown pymdown-extensions playwright && playwright install chromium
    weasyprint(可选):  pip install weasyprint

特性:
    - 支持中文 (内置中文字体配置)
    - 支持 LaTeX 数学公式 ($...$, $$...$$), 经 MathJax 渲染 (playwright 后端)
    - 自动将本地图片转为 base64 嵌入 (无需网络)
    - 支持 GitHub Flavored Markdown (表格、代码块等)
    - 可选自定义 CSS 样式
"""

import argparse
import base64
import os
import re
import sys
import markdown


# MathJax v3 配置: 配合 pymdownx.arithmatex (generic=True) 输出的 \(...\) / \[...\]
# pageReady 在初始排版完成后置 mathJaxReady=true, 供 playwright 轮询等待
MATHJAX_HEAD = r"""<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true
  },
  startup: {
    pageReady: () => MathJax.startup.defaultPageReady().then(() => { window.mathJaxReady = true; })
  }
};
</script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def image_to_base64(image_path):
    """将图片文件转为 base64 data URI"""
    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    mime_map = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'gif': 'image/gif', 'svg': 'image/svg+xml', 'bmp': 'image/bmp',
                'webp': 'image/webp'}
    mime = mime_map.get(ext, 'image/png')
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:{mime};base64,{data}'


def embed_images(html, base_dir):
    """将 HTML 中的本地图片路径替换为 base64 编码"""
    def replace_img(match):
        src = match.group(1)
        # 跳过已经是 data URI 的、http 链接的、或锚点
        if src.startswith('data:') or src.startswith('http') or src.startswith('#'):
            return match.group(0)
        # 处理路径: 绝对路径直接用, 相对路径基于 md 文件所在目录
        if os.path.isabs(src):
            img_path = src
        else:
            img_path = os.path.normpath(os.path.join(base_dir, src))
        if os.path.exists(img_path):
            data_uri = image_to_base64(img_path)
            return match.group(0).replace(src, data_uri)
        else:
            print(f"  警告: 图片不存在: {img_path}", file=sys.stderr)
            return match.group(0)

    # 匹配 <img src="..."> 和 <img src='...'>
    html = re.sub(r'<img[^>]+src=["\']([^"\']+)["\']', replace_img, html)
    return html


def generate_css():
    """生成默认 CSS 样式 (含中文字体支持)"""
    return """
<style>
@page {
    size: A4;
    margin: 2cm 1.5cm;
}

body {
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                 "Microsoft YaHei", "WenQuanYi Micro Hei", "SimHei",
                 "DejaVu Sans", "Arial", sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

h1 {
    font-size: 20pt;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 8px;
    margin-top: 20px;
}

h2 {
    font-size: 16pt;
    color: #222;
    border-bottom: 1px solid #999;
    padding-bottom: 5px;
    margin-top: 18px;
}

h3 {
    font-size: 13pt;
    color: #333;
    margin-top: 15px;
}

h4 {
    font-size: 11pt;
    color: #444;
}

p {
    margin: 8px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
}

th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
}

th {
    background-color: #f5f5f5;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

code {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", "Courier New", monospace;
    background-color: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
}

pre {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    font-size: 9.5pt;
    line-height: 1.4;
}

pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
}

blockquote {
    border-left: 4px solid #4CAF50;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #f9f9f9;
    color: #666;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 20px 0;
}

a {
    color: #1976D2;
    text-decoration: none;
}

ul, ol {
    padding-left: 24px;
    margin: 8px 0;
}

li {
    margin: 4px 0;
}
</style>
"""


def render_pdf_playwright(full_html, pdf_path):
    """用 playwright + MathJax 渲染 HTML 为 PDF (支持 LaTeX 公式)"""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        # root 环境需 --no-sandbox; --disable-dev-shm-usage 避免 /dev/shm 太小
        browser = p.chromium.launch(args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        page.set_content(full_html, wait_until='networkidle', timeout=120000)
        # 等待 MathJax 初始排版完成 (pageReady 置 mathJaxReady=true)
        try:
            page.wait_for_function("() => window.mathJaxReady === true", timeout=60000)
        except Exception as e:
            print(f"  警告: MathJax 渲染等待超时, 公式可能未渲染: {e}", file=sys.stderr)
        page.pdf(
            path=pdf_path,
            format='A4',
            margin={'top': '2cm', 'bottom': '2cm', 'left': '1.5cm', 'right': '1.5cm'},
            print_background=True,
        )
        browser.close()


def md_to_pdf(md_path, pdf_path=None, css_path=None, backend='playwright'):
    """将 Markdown 文件转换为 PDF"""
    if pdf_path is None:
        pdf_path = os.path.splitext(md_path)[0] + '.pdf'

    md_dir = os.path.dirname(os.path.abspath(md_path))

    # 读取 Markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 转换为 HTML: playwright 后端启用 arithmatex 保护 $...$/$$...$$ 公式
    extensions = ['tables', 'fenced_code', 'codehilite',
                  'toc', 'smarty', 'sane_lists']
    extension_configs = {}
    use_math = (backend == 'playwright')
    if use_math:
        extensions.append('pymdownx.arithmatex')
        extension_configs['pymdownx.arithmatex'] = {'generic': True}

    html_body = markdown.markdown(
        md_text,
        extensions=extensions,
        extension_configs=extension_configs,
    )

    # 嵌入图片
    html_body = embed_images(html_body, md_dir)

    # 加载 CSS
    if css_path and os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f'<style>\n{f.read()}\n</style>'
    else:
        css_content = generate_css()

    head_extra = MATHJAX_HEAD if use_math else ''

    # 组装完整 HTML
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
{css_content}
{head_extra}
</head>
<body>
{html_body}
</body>
</html>"""

    # 生成 PDF
    if backend == 'playwright':
        render_pdf_playwright(full_html, pdf_path)
    else:
        from weasyprint import HTML
        HTML(string=full_html, base_url=md_dir).write_pdf(pdf_path)

    file_size = os.path.getsize(pdf_path) / 1024
    print(f"PDF 已生成: {pdf_path} ({file_size:.1f} KB)")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description='Markdown 转 PDF 工具 (支持中文、本地图片嵌入)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python md2pdf.py report.md
  python md2pdf.py report.md -o output.pdf
  python md2pdf.py report.md --css custom.css
  python md2pdf.py report.md --backend weasyprint   # 无浏览器离线兜底, 不渲染公式
        """
    )
    parser.add_argument('input', help='输入 Markdown 文件路径')
    parser.add_argument('-o', '--output', default=None, help='输出 PDF 路径 (默认同名 .pdf)')
    parser.add_argument('--css', default=None, help='自定义 CSS 样式文件路径')
    parser.add_argument('--backend', default='playwright', choices=['playwright', 'weasyprint'],
                        help='渲染后端: playwright(默认,支持公式) / weasyprint(离线兜底,不渲染公式)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    md_to_pdf(args.input, args.output, args.css, args.backend)


if __name__ == '__main__':
    main()
