# yoDEV Course Library

Curated courses for Latin American developers, available for import into yoDEV Workplace.

## Available Courses

| Course | Language | Open in Workplace |
|--------|----------|-------------------|
| [AI Agents for Beginners](courses/ai-agents-es/) | Spanish | [**Abrir**](https://workplace.yodev.dev/template/import?name=Agentes%20de%20IA%20para%20Principiantes&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ai-agents-es/course.zip) |
| [Hugging Face NLP Course](courses/hf-nlp-es/) | Spanish | [**Abrir**](https://workplace.yodev.dev/template/import?name=Curso%20de%20NLP%20de%20Hugging%20Face&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/hf-nlp-es/course.zip) |

---

## Course Details

### 1. AI Agents for Beginners (Spanish)

- **Source:** [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- **Content:** 15 lessons on AI agent development
- **Topics:** Agent frameworks, tool use, RAG, multi-agent patterns, memory
- **Package:** ~2.7MB (with base64 embedded images)

### 2. Hugging Face NLP Course (Spanish)

- **Source:** [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- **Content:** Chapters 0-8 (Part 1: Fundamentals)
- **Topics:** Transformers, tokenization, fine-tuning, datasets, NLP tasks
- **Package:** ~494KB

---

## How It Works

1. Click "Abrir" link for any course
2. Sign in with your yoDEV account (if not already)
3. Select target workspace
4. Course is imported as an editable document

See [docs/authentication-flow.md](docs/authentication-flow.md) for technical details.

---

## Repository Structure

```
workplace-course-integration/
├── courses/
│   ├── ai-agents-es/         # Microsoft AI Agents course
│   │   ├── course.zip        # BlockSuite snapshot
│   │   ├── source/           # Source markdown
│   │   └── README.md
│   └── hf-nlp-es/            # Hugging Face NLP course
│       ├── course.zip        # BlockSuite snapshot
│       ├── source/           # Source markdown
│       └── README.md
├── docs/
│   ├── authentication-flow.md
│   ├── image-handling.md
│   └── technical-guide.md
├── scripts/                   # Processing scripts
└── templates/                 # API specs
```

## Adding a New Course

1. Compile/translate course content to markdown
2. Import markdown into Workplace
3. Export as Snapshot (Document menu → Export → Snapshot)
4. Upload `course.zip` to R2: `courses/{course-id}/course.zip`
5. Add entry to this README with import link
6. Create course README in `courses/{course-id}/`

## R2 Storage

Snapshots are stored in Cloudflare R2:
```
https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/{course-id}/course.zip
```

## Related

- [yoDEV Workplace](https://workplace.yodev.dev) - AFFiNE-based workspace
- [yoDEV Community](https://yodev.dev) - Discourse forums

## License

MIT
