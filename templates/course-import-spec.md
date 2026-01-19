# Course Import API Specification (Template)

## Overview

This template outlines the API design for the "Open in Workplace" feature.

## Endpoints

### POST /api/import/prepare

Prepares a course for import into user's workspace.

**Request:**
```json
{
  "courseId": "ai-agents-es",
  "userId": "user-uuid",
  "workspaceId": "workspace-uuid" // optional, creates new if omitted
}
```

**Response:**
```json
{
  "importToken": "token-uuid",
  "expiresAt": "2024-01-20T12:00:00Z",
  "courseMetadata": {
    "title": "Agentes de IA para Principiantes",
    "lessons": 15,
    "estimatedSize": "7.5MB"
  }
}
```

### GET /api/import/content

Retrieves course content for import.

**Request:**
```
GET /api/import/content?token=token-uuid
Authorization: Bearer [user-token]
```

**Response:**
```json
{
  "markdown": "# Course content...",
  "metadata": {
    "title": "...",
    "source": "...",
    "version": "1.0"
  }
}
```

### POST /api/import/complete

Marks import as complete (for analytics/tracking).

**Request:**
```json
{
  "importToken": "token-uuid",
  "documentId": "created-doc-uuid",
  "status": "success"
}
```

## Authentication

- Uses existing OIDC flow between Discourse and Workplace
- Import tokens are short-lived (5 minutes)
- User must be authenticated before import

## Error Handling

| Code | Description |
|------|-------------|
| 401 | Not authenticated |
| 403 | No access to course |
| 404 | Course not found |
| 410 | Import token expired |
| 429 | Rate limit exceeded |

## Rate Limits

- 10 imports per user per hour
- 100 imports total per hour (system-wide)

## Implementation Notes

*To be completed during implementation*
