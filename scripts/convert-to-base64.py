#!/usr/bin/env python3
"""Convert markdown images to base64 data URIs"""

import re
import base64
from pathlib import Path

# Mapping of R2 URLs to local files
R2_BASE = "https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ai-agents-es/images/"

def get_mime_type(filename):
    if filename.endswith('.webp'):
        return 'image/webp'
    elif filename.endswith('.svg'):
        return 'image/svg+xml'
    elif filename.endswith('.png'):
        return 'image/png'
    return 'image/webp'

def convert_image_to_base64(local_path):
    """Convert image file to base64 data URI"""
    with open(local_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    mime = get_mime_type(str(local_path))
    return f"data:{mime};base64,{data}"

def process_markdown(input_file, output_file, images_dir):
    """Replace image URLs with base64 data URIs"""
    with open(input_file, 'r') as f:
        content = f.read()

    # Find all markdown images
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_image(match):
        alt_text = match.group(1)
        url = match.group(2)

        # Extract filename from URL
        if R2_BASE in url:
            filename = url.replace(R2_BASE, '')
            local_path = images_dir / filename

            if local_path.exists():
                data_uri = convert_image_to_base64(local_path)
                print(f"  Converted: {filename}")
                return f"![{alt_text}]({data_uri})"
            else:
                print(f"  WARNING: Not found: {filename}")
                return match.group(0)
        else:
            print(f"  Skipping external URL: {url[:50]}...")
            return match.group(0)

    print("Converting images to base64...")
    converted = re.sub(pattern, replace_image, content)

    with open(output_file, 'w') as f:
        f.write(converted)

    print(f"\nOutput written to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    base_dir = Path('/Users/grego/ai-agents-course-images')
    process_markdown(
        base_dir / 'ai-agents-course-es-with-videos-no-toc.md',
        base_dir / 'ai-agents-course-es-final-v2.md',
        base_dir / 'images'
    )
