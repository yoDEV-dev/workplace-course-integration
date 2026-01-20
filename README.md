# yoDEV Course Library

Curated courses for Latin American developers, available for import into yoDEV Workplace.

## Available Courses

### AI Agents for Beginners (Spanish)
| | |
|---|---|
| **15 lecciones** | [**Abrir en Workplace**](https://workplace.yodev.dev/template/import?name=Agentes%20de%20IA%20para%20Principiantes&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ai-agents-es/course.zip) |

### Hugging Face NLP Course (Spanish)
| Parte | Contenido | |
|-------|-----------|---|
| Parte 1 | Fundamentos (Caps 0-4) | [**Abrir**](https://workplace.yodev.dev/template/import?name=HF%20NLP%20Parte%201%20-%20Fundamentos&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/hf-nlp-es/part1-fundamentos.zip) |
| Parte 2 | Bibliotecas (Caps 5-6) | [**Abrir**](https://workplace.yodev.dev/template/import?name=HF%20NLP%20Parte%202%20-%20Bibliotecas&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/hf-nlp-es/part2-bibliotecas.zip) |
| Parte 3 | Aplicaciones (Caps 7-8) | [**Abrir**](https://workplace.yodev.dev/template/import?name=HF%20NLP%20Parte%203%20-%20Aplicaciones&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/hf-nlp-es/part3-aplicaciones.zip) |

### ML para Principiantes (Spanish)
| Parte | Contenido | |
|-------|-----------|---|
| Parte 1 | Fundamentos (Intro + Regresión) | [**Abrir**](https://workplace.yodev.dev/template/import?name=ML%20Parte%201%20-%20Fundamentos&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ml-beginners-es/ml-beginners-part1-fundamentos.zip) |
| Parte 2 | Clasificación (Web + Classification + Clustering) | [**Abrir**](https://workplace.yodev.dev/template/import?name=ML%20Parte%202%20-%20Clasificacion&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ml-beginners-es/ml-beginners-part2-clasificacion.zip) |
| Parte 3 | NLP (Procesamiento de Lenguaje Natural) | [**Abrir**](https://workplace.yodev.dev/template/import?name=ML%20Parte%203%20-%20NLP&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ml-beginners-es/ml-beginners-part3-nlp.zip) |
| Parte 4 | Avanzado (Time Series + RL + Real-World) | [**Abrir**](https://workplace.yodev.dev/template/import?name=ML%20Parte%204%20-%20Avanzado&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ml-beginners-es/ml-beginners-part4-avanzado.zip) |

---

## Course Details

### 1. AI Agents for Beginners
- **Source:** [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- **Content:** 15 lessons on AI agent development
- **Topics:** Agent frameworks, tool use, RAG, multi-agent patterns, memory

### 2. Hugging Face NLP Course
- **Source:** [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- **Content:** Chapters 0-8 (70 sections)
- **Topics:** Transformers, tokenization, fine-tuning, datasets, NLP tasks

### 3. ML para Principiantes
- **Source:** [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners)
- **Content:** 26 lessons in 4 parts
- **Topics:** Regression, classification, clustering, NLP, time series, reinforcement learning

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
│   ├── ai-agents-es/
│   ├── hf-nlp-es/
│   └── ml-beginners-es/
├── docs/
├── scripts/
└── templates/
```

## Adding a New Course

1. Compile/translate course content to markdown
2. Import markdown into Workplace
3. Export as Snapshot (Document menu → Export → Snapshot)
4. Upload snapshot to R2: `courses/{course-id}/`
5. Add entry to this README
6. Create course README in `courses/{course-id}/`

## R2 Storage

```
https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/{course-id}/
```

## Related

- [yoDEV Workplace](https://workplace.yodev.dev)
- [yoDEV Community](https://yodev.dev)

## License

MIT
