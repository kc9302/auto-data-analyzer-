import re
import xml.etree.ElementTree as ET

with open('docs/system_architecture.html', 'r', encoding='utf-8') as f:
    html = f.read()

m = re.search(r'<svg[\s\S]*?</svg>', html, re.IGNORECASE)
if not m:
    print("Could not find SVG in HTML")
    exit(1)

raw_svg = m.group(0)

# Add SVG namespace and standard attributes
if 'xmlns="http://www.w3.org/2000/svg"' not in raw_svg:
    raw_svg = raw_svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ')

# Extract CSS from html
css_matches = re.findall(r'<style[^>]*>([\s\S]*?)</style>', html)
full_css = '\n'.join(css_matches)

# Inject <style> inside <defs> with CDATA to prevent XML parser collision
defs_replacement = f'''<defs>
    <style type="text/css"><![CDATA[
{full_css}
    ]]></style>'''

clean_svg = raw_svg.replace('<defs>', defs_replacement)

# Add solid background rect so GitHub light/dark mode doesn't produce transparent background
if '<rect width="100%" height="100%" fill="#020617"/>' not in clean_svg:
    clean_svg = clean_svg.replace('<rect width="100%" height="100%" fill="url(#grid)" />', '<rect width="100%" height="100%" fill="#020617"/><rect width="100%" height="100%" fill="url(#grid)" />')

# Validate XML before writing
try:
    ET.fromstring(clean_svg)
    print("Verification: SVG is 100% VALID XML!")
except Exception as e:
    print(f"Verification FAILED: {e}")
    exit(1)

with open('docs/system_architecture.svg', 'w', encoding='utf-8') as f:
    f.write(clean_svg)

print(f"Successfully saved docs/system_architecture.svg (Length: {len(clean_svg)} bytes)")
