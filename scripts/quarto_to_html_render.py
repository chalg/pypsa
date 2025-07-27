# Script to render Quarto document to HTML and move files
# This is because the yaml header in the Quarto document is unable to output to a specific directory (for a single file), so the HTML file is saved to the root directory.
# Note: I'm rendering a single html file, which I'm renaming and moving, I'm not creating a full website and my project type is not Website Project.

import subprocess
import shutil
import os

# Render the file
result = subprocess.run([
    "quarto", "render", "High-level_NEM.qmd", 
    "--to", "html"
], capture_output=True, text=True)

# Create docs directory if it doesn't exist
os.makedirs("docs", exist_ok=True)

# Move files if rendering was successful (yaml only saves to root directory for some reason)
if result.returncode == 0:
    # Move and rename the HTML file
    shutil.move("index.html", "docs/index.html")
    print("✅ File moved to docs/index.html")
    
    # Move the supporting files folder if it exists
    if os.path.exists("High-level_NEM_files"):
        # Remove existing folder in docs if it exists
        if os.path.exists("docs/High-level_NEM_files"):
            shutil.rmtree("docs/High-level_NEM_files")
        
        shutil.move("High-level_NEM_files", "docs/High-level_NEM_files")
        print("✅ Supporting files folder moved to docs/High-level_NEM_files")
    else:
        print("No supporting files folder found")
        
else:
    print(f"❌ Render failed:\n{result.stderr}")