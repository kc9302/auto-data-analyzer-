"""
Web UI Launcher Shortcut
Usage: uv run streamlit run web.py
"""
import os
import sys

# Forward directly to src/web/app.py
app_path = os.path.join(os.path.dirname(__file__), "src", "web", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, 'exec')
    exec(code, globals())
