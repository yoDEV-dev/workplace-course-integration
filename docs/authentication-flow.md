# Authentication Flow

## Overview

This document describes the "Open in Workplace" authentication and document cloning flow.

**Status:** ✅ Implemented and tested

## User Flow

```
1. User clicks "Open in Workplace" link on yoDEV.dev (Discourse)
2. Browser navigates to Workplace import URL
3. Workplace checks authentication:
   - If not logged in → Redirect to OIDC sign-in → Return to import URL
   - If logged in → Continue
4. Workspace selector dialog appears
5. User selects target workspace (or creates new)
6. Course ZIP downloaded from R2 and imported
7. User redirected to imported document in their workspace
```

## Implementation

### No Custom Code Required

Workplace has built-in template import functionality at `/template/import` that handles:
- Authentication checks with OIDC redirect
- ZIP download from external URL
- Workspace selection UI
- Document import via `ZipTransformer`
- Redirect to imported document

### Import URL Format

```
https://workplace.yodev.dev/template/import?name={NAME}&mode={MODE}&snapshotUrl={URL}
```

**Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Display name for the template (URL encoded) |
| `mode` | Yes | Document mode: `page` or `edgeless` |
| `snapshotUrl` | Yes | Full URL to the course ZIP file |

### Example URL

```
https://workplace.yodev.dev/template/import?name=Agentes%20de%20IA%20para%20Principiantes&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ai-agents-es/course.zip
```

## Course Storage

### R2 Bucket Structure

```
courses/
└── ai-agents-es/
    ├── course.zip          # BlockSuite snapshot (3.5MB)
    └── images/             # Original images (backup, not used for import)
```

### ZIP Format

The `course.zip` file is a BlockSuite snapshot export containing:
- `*.snapshot.json` - Document structure and content
- `assets/` - Embedded images with hash-based filenames

Export via Workplace: Document menu → Export → Snapshot

### CORS Configuration

R2 bucket must allow requests from Workplace origin.

**Cloudflare R2 CORS Policy:**
```json
[
  {
    "AllowedOrigins": [
      "https://workplace.yodev.dev"
    ],
    "AllowedMethods": [
      "GET",
      "HEAD"
    ],
    "AllowedHeaders": [
      "*"
    ],
    "MaxAgeSeconds": 3600
  }
]
```

## OIDC Integration

The existing OIDC bridge handles authentication:

1. Workplace detects unauthenticated user
2. Redirects to sign-in with return URL preserved
3. OIDC bridge authenticates via Discourse SSO
4. User returns to original import URL
5. Import proceeds with authenticated session

**Key:** The `snapshotUrl` parameter is preserved through the auth redirect automatically.

## Technical Components

### Workplace (AFFiNE Fork)

**Files involved:**
- `/packages/frontend/core/src/desktop/pages/import-template/index.tsx` - Route handler
- `/packages/frontend/core/src/desktop/dialogs/import-template/index.tsx` - Import dialog
- `/packages/frontend/core/src/modules/import-template/services/import.ts` - Import service

**Key service:**
```typescript
ImportTemplateService.importToWorkspace(workspaceMetadata, docBinary, mode)
```

### OIDC Bridge

- Existing `discourse-oidc-bridge` handles Discourse ↔ Workplace authentication
- No modifications needed for course import

## Adding New Courses

1. Create course document in Workplace
2. Export as Snapshot (ZIP)
3. Upload to R2: `courses/{course-id}/course.zip`
4. Create import URL with appropriate `name` and `snapshotUrl`
5. Add link to Discourse (catalogue, theme component, etc.)

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| CORS error | R2 bucket missing CORS policy | Add `AllowedOrigins` for Workplace domain |
| 404 on snapshotUrl | ZIP not uploaded or wrong path | Verify R2 path matches URL |
| Auth redirect loop | OIDC misconfiguration | Check OIDC bridge logs |
| Import fails | Corrupted ZIP or invalid format | Re-export from Workplace |
