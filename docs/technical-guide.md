# Technical Guide

## Architecture Overview

### Components

1. **Source Content** - External course repositories (e.g., Microsoft GitHub)
2. **R2 Storage** - Cloudflare R2 for image hosting
3. **Workplace** - yoDEV's AFFiNE fork for document viewing/editing
4. **Discourse** - Community platform with OIDC authentication

### Data Flow

```
Source Repo → Compile Content → Process Images → Import to Workplace → User Access
                    ↓                  ↓
              Translate (if       Base64 embed
              needed)             or R2 URLs
```

## Workplace/AFFiNE Technical Details

### Document Storage Model

AFFiNE uses **Yjs CRDT** for document storage:
- Documents stored as binary blobs, not raw markdown
- Images stored internally as blobs with `sourceId` references
- GraphQL API at `/graphql`
- Prisma ORM with models: `Workspace`, `WorkspaceDoc`, `Snapshot`, `Update`

### BlockSuite Editor

The editor uses BlockSuite with specific adapters for format conversion:
- Location: `packages/blocks/src/_common/adapters/markdown/`
- Key APIs: `importMarkdownToDoc`, `MarkdownAdapter`
- Conversion happens client-side (requires browser environment)

### Known Limitations

1. **Markdown anchor links** (`#section-name`) don't work for internal navigation
   - Solution: Use Workplace's native TOC feature (sidebar/popup)

2. **Markdown image syntax** (`![alt](url)`) may not convert to image blocks
   - External URLs require image proxy configuration
   - Solution: Embed images as base64 data URIs

3. **Large file imports** cause browser unresponsiveness
   - Expected behavior for files >1MB
   - Browser "wait" option completes successfully

### Image Proxy (Self-Hosted)

For external image URLs to work, configure:
```javascript
imageProxyUrl: '/api/worker/image-proxy'
linkPreviewUrl: '/api/worker/link-preview'
```

## Import Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| MCP Document Creation | Programmatic, automatable | Markdown may not render correctly |
| Manual Paste (Import Tool) | Better rendering, tables work | Manual process |
| Base64 Embedded Images | Reliable image rendering | Large file sizes, slow import |
| External URL Images | Small file sizes | Requires proxy config, may not render |

**Recommended:** Base64 embedded images with manual import tool for best reliability.

## Content Processing Pipeline

### Step 1: Compile Course Content

Extract and combine lesson content from source repository:
- Pull README files from each lesson directory
- Maintain heading hierarchy
- Preserve code blocks and tables
- Extract image references

### Step 2: Process Images

Two approaches:

**A. R2 Storage (External URLs)**
```bash
# Download images
curl -sL "[source-url]" -o "images/[filename]"

# Upload to R2
wrangler r2 object put yodev-assets/courses/[course]/images/[file] --file ./images/[file]
```

**B. Base64 Embedding (Recommended)**
```python
import base64

def convert_image_to_base64(path):
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    mime = 'image/webp'  # or image/png, image/svg+xml
    return f"data:{mime};base64,{data}"
```

### Step 3: Add Video Links

For courses with video content:
```python
VIDEOS = {
    "01": "https://youtu.be/xxxxx",
    # ...
}
# Insert after lesson thumbnail
```

Note: Workplace has YouTube embed feature - use `/embed` command for inline video players.

### Step 4: Remove TOC

Workplace has native TOC (sidebar/popup), so markdown TOC is redundant:
```bash
sed '/^## Tabla de Contenidos$/,/^---$/d' input.md > output.md
```

### Step 5: Import to Workplace

1. Open Workplace
2. Create new document or workspace
3. Use markdown import function
4. Wait for processing (may show "unresponsive" for large files)
5. Verify images and formatting

## File Size Considerations

| Content Type | Approximate Size |
|--------------|-----------------|
| Text + Code (15 lessons) | ~50 KB |
| Single image (base64) | 100-400 KB |
| Full course (53 images) | ~7.5 MB |

Import time scales with file size. Expect several minutes for full course.

## Future: Authentication Flow

Planned implementation for "Open in Workplace" button:

```
User clicks button → OIDC auth check → Clone document to user workspace → Redirect to Workplace
```

Technical details to be documented as implementation progresses.
