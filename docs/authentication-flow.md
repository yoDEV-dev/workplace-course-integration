# Authentication Flow (Planned)

## Overview

This document will describe the "Open in Workplace" authentication and document cloning flow.

**Status:** Planning phase - implementation pending

## Proposed User Flow

```
1. User browses course catalogue on yoDEV.dev (Discourse)
2. User clicks "Open in Workplace" button
3. System checks OIDC authentication status
   - If not authenticated: Redirect to login, then continue
   - If authenticated: Proceed
4. System clones course document to user's Workspace
5. User redirected to Workplace with document open
6. User can edit, annotate, and get AI assistance
```

## Technical Components

### 1. Course Catalogue (Discourse)

- Course listing with metadata
- "Open in Workplace" button/card
- Authentication state awareness

### 2. Import API (Workplace Backend)

Proposed endpoints:

```
POST /api/import/prepare
- Creates time-limited import token
- Returns: { token, expiresAt }

GET /api/import/content?token=xxx
- Returns course markdown + metadata
- Validates token

POST /api/import/complete
- Marks import as successful
- Analytics tracking
```

### 3. Document Cloning

Options to investigate:
- Clone existing document to user's workspace
- Import from template on-demand
- Pre-provisioned workspaces with course content

### 4. OIDC Integration

- Existing: Discourse ↔ Workplace OIDC
- Ensure seamless handoff during import flow
- Handle session expiry gracefully

## Design Considerations

### User Experience
- Minimize clicks from catalogue to document
- Clear feedback during cloning process
- Handle errors gracefully (network issues, auth failures)

### Performance
- Consider pre-loading/caching for popular courses
- Async document creation if needed

### Security
- Validate user permissions
- Rate limiting on clone operations
- Audit logging for compliance

## Implementation Notes

*To be added as implementation progresses*

## References

- [yoDEV Workplace Repo](https://github.com/yoDEV-dev/affine-yodev-prod)
- [AFFiNE GraphQL API](https://docs.affine.pro/)
- [BlockSuite Documentation](https://block-suite.com/)
