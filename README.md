# Workplace Course Integration

Documentation and tools for integrating educational courses into yoDEV Workplace (AFFiNE fork).

## Overview

This project enables yoDEV to offer curated, translated courses to Latin American developers through their Workplace platform. Users can open courses in their personal Workspace with full editing capabilities and AI assistance.

## Current Status

### Completed
- [x] Course content compilation (Microsoft AI Agents for Beginners - Spanish)
- [x] Image migration to R2 storage
- [x] Base64 image embedding for Workplace import
- [x] Video embedding (YouTube)
- [x] Workplace import testing and optimization

### In Progress
- [ ] Course catalogue design and navigation
- [ ] "Open in Workplace" button implementation
- [ ] OIDC authentication flow
- [ ] Document cloning to user workspaces

## Project Structure

```
workplace-course-integration/
├── README.md                     # This file
├── docs/
│   ├── technical-guide.md        # Technical implementation details
│   ├── image-handling.md         # Image processing documentation
│   └── authentication-flow.md    # OIDC integration (planned)
├── scripts/
│   ├── download-images.sh        # Download course images
│   ├── convert-to-base64.py      # Embed images as base64
│   └── add-videos.py             # Add video links to lessons
├── courses/
│   └── ai-agents-es/
│       ├── image-mapping.json    # Original to R2 URL mapping
│       └── README.md             # Course-specific notes
└── templates/
    └── course-import-spec.md     # API specification template
```

## Quick Start

### Prerequisites
- Python 3.12+
- GitHub CLI (`gh`)
- Access to yoDEV R2 bucket
- Workplace instance access

### Processing a New Course

1. **Download images** from source repository
2. **Upload to R2** (optional - for external URL approach)
3. **Convert to base64** for reliable Workplace import
4. **Add video links** if course has video content
5. **Import via Workplace** markdown import function

## First Course: AI Agents for Beginners (Spanish)

- **Source:** [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- **Lessons:** 15 (00-14)
- **Images:** 53
- **Videos:** 13 (YouTube embedded)
- **Language:** Spanish (machine-translated from Microsoft repo)

## Related Resources

- [yoDEV Workplace](https://github.com/yoDEV-dev/affine-yodev-prod) - AFFiNE fork
- [AFFiNE](https://github.com/toeverything/AFFiNE) - Upstream project
- [BlockSuite](https://github.com/toeverything/blocksuite) - Editor framework

## License

MIT
