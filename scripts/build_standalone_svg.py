import re

with open('docs/system_architecture.html', 'r', encoding='utf-8') as f:
    html = f.read()

css_matches = re.findall(r'<style[^>]*>([\s\S]*?)</style>', html)
diagram_css = ''
for s in css_matches:
    if 'c-region' in s or 'c-external' in s or 'a-default' in s or 'theme-dark' in s:
        diagram_css += s + '\n'

# Default to dark theme colors for github dark/light compatibility or crisp layout
# Let's inspect root variables in html
root_vars = ''
m_root = re.search(r':root\s*\{[\s\S]*?\}', html)
if m_root:
    root_vars += m_root.group(0) + '\n'
m_dark = re.search(r'\[data-theme="dark"\]\s*\{[\s\S]*?\}', html)
if m_dark:
    root_vars += m_dark.group(0) + '\n'

with open('docs/system_architecture.svg', 'r', encoding='utf-8') as f:
    svg_content = f.read()

if 'xmlns="http://www.w3.org/2000/svg"' not in svg_content:
    svg_content = svg_content.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ')

style_block = f'<style>\n{root_vars}\n{diagram_css}\n</style>'
svg_content = svg_content.replace('<!-- Definitions -->', f'<!-- Definitions -->\n{style_block}')

# Ensure background rect is solid dark
if '<rect width="100%" height="100%" fill="#020617"/>' not in svg_content:
    svg_content = svg_content.replace('<rect width="100%" height="100%" fill="url(#grid)" />', '<rect width="100%" height="100%" fill="#020617"/><rect width="100%" height="100%" fill="url(#grid)" />')

with open('docs/system_architecture.svg', 'w', encoding='utf-8') as f:
    f.write(svg_content)

print("Saved self-contained SVG successfully!")
