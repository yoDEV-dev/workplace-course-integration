# Image Handling Guide

## The Challenge

AFFiNE/BlockSuite uses a **blob-based storage model** for images:
- Images are stored internally, not as external URLs
- Standard markdown `![alt](url)` syntax may not convert to visual images
- Self-hosted instances need specific proxy configuration

## Solutions

### Solution 1: Base64 Data URIs (Recommended)

Embed images directly in the markdown as base64-encoded data URIs.

**Pros:**
- Reliable rendering in Workplace import
- No external dependencies
- Works without image proxy configuration

**Cons:**
- Large file sizes (~33% larger than binary)
- Slow import for many images
- Browser may become unresponsive during import

**Implementation:**

```python
import base64
import re
from pathlib import Path

def get_mime_type(filename):
    if filename.endswith('.webp'):
        return 'image/webp'
    elif filename.endswith('.svg'):
        return 'image/svg+xml'
    elif filename.endswith('.png'):
        return 'image/png'
    return 'image/webp'

def convert_image_to_base64(local_path):
    with open(local_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    mime = get_mime_type(str(local_path))
    return f"data:{mime};base64,{data}"

def process_markdown(input_file, output_file, images_dir):
    with open(input_file, 'r') as f:
        content = f.read()

    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_image(match):
        alt_text = match.group(1)
        url = match.group(2)

        # Extract filename and find local file
        filename = url.split('/')[-1]
        local_path = images_dir / filename

        if local_path.exists():
            data_uri = convert_image_to_base64(local_path)
            return f"![{alt_text}]({data_uri})"
        return match.group(0)

    converted = re.sub(pattern, replace_image, content)

    with open(output_file, 'w') as f:
        f.write(converted)
```

### Solution 2: R2 External URLs

Store images in Cloudflare R2 and reference via public URLs.

**Pros:**
- Small markdown file sizes
- Images can be updated without re-importing document

**Cons:**
- May not render without image proxy configuration
- Dependency on external storage availability

**R2 Setup:**

1. Create directory structure:
```
courses/
└── [course-name]/
    └── images/
        ├── lesson-01-thumbnail.webp
        ├── lesson-01-diagram.webp
        └── ...
```

2. Upload via Cloudflare dashboard or wrangler CLI:
```bash
wrangler r2 object put bucket-name/courses/ai-agents-es/images/file.webp --file ./file.webp
```

3. Public URL format:
```
https://pub-[bucket-id].r2.dev/courses/[course]/images/[filename]
```

**Image Mapping JSON:**

Track original URLs to R2 URLs:
```json
{
  "r2BaseUrl": "https://pub-xxx.r2.dev/courses/ai-agents-es/images",
  "images": [
    {
      "localFile": "lesson-01-thumbnail.webp",
      "r2Url": "https://pub-xxx.r2.dev/courses/ai-agents-es/images/lesson-01-thumbnail.webp",
      "originalUrl": "https://github.com/microsoft/repo/images/thumbnail.webp"
    }
  ]
}
```

### Solution 3: Manual /image Command

Use Workplace's built-in `/image` command to insert images after import.

**Pros:**
- Guaranteed to work
- Images properly stored in blob system

**Cons:**
- Manual process for each image
- Time-consuming for courses with many images

**Workflow:**
1. Import markdown without images (or with placeholder text)
2. Navigate to each image location
3. Type `/image` and select/upload from URL
4. Repeat for each image

## Image Download Script

For downloading images from source repositories:

```bash
#!/bin/bash

OUTPUT_DIR="./images"
mkdir -p "$OUTPUT_DIR"

# Example: Microsoft AI Agents course
BASE_URL="https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es"

# Download each image
curl -sL "${BASE_URL}/lesson-1-thumbnail.webp" -o "$OUTPUT_DIR/lesson-01-thumbnail.webp"
curl -sL "${BASE_URL}/what-are-ai-agents.webp" -o "$OUTPUT_DIR/lesson-01-what-are-ai-agents.webp"
# ... continue for all images

echo "Downloaded $(ls -1 $OUTPUT_DIR | wc -l) images"
```

## Supported Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| WebP | image/webp | Recommended - good compression |
| PNG | image/png | Lossless, larger files |
| SVG | image/svg+xml | Vector graphics, may have rendering issues |
| JPEG | image/jpeg | Good for photos |

## Troubleshooting

### Images show as text links
- Use base64 embedding instead of external URLs
- Check image proxy configuration if using external URLs

### Import hangs or browser unresponsive
- Normal for large files (>5MB)
- Click "Wait" when browser prompts
- Consider splitting into smaller documents

### SVG images don't render
- Some complex SVGs may not render correctly
- Convert to PNG/WebP as fallback

### Base64 encoding errors
- Ensure binary read mode (`'rb'`)
- Verify MIME type matches file extension
