# AI Agents for Beginners (Spanish)

## Course Information

- **Original Source:** [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- **Language:** Spanish (machine-translated from Microsoft repo)
- **Lessons:** 15 (00-14)
- **Images:** 53
- **Videos:** 13 (YouTube, Lessons 01-13)

## Content Summary

| Lesson | Title | Video |
|--------|-------|-------|
| 00 | Configuración del Curso | - |
| 01 | Introducción a los Agentes de IA | [YouTube](https://youtu.be/3zgm60bXmQk) |
| 02 | Explorar Marcos de Agentes de IA | [YouTube](https://youtu.be/ODwF-EZo_O8) |
| 03 | Principios de Diseño de Agentes | [YouTube](https://youtu.be/m9lM8qqoOEA) |
| 04 | Patrón de Uso de Herramientas | [YouTube](https://youtu.be/vieRiPRx-gI) |
| 05 | Agentic RAG | [YouTube](https://youtu.be/WcjAARvdL7I) |
| 06 | Construcción de Agentes Confiables | [YouTube](https://youtu.be/iZKkMEGBCUQ) |
| 07 | Diseño de Planificación | [YouTube](https://youtu.be/kPfJ2BrBCMY) |
| 08 | Patrones Multi-Agente | [YouTube](https://youtu.be/V6HpE9hZEx0) |
| 09 | Metacognición | [YouTube](https://youtu.be/His9R6gw6Ec) |
| 10 | Agentes en Producción | [YouTube](https://youtu.be/l4TP6IyJxmQ) |
| 11 | Protocolos Agénticos | [YouTube](https://youtu.be/X-Dh9R3Opn8) |
| 12 | Ingeniería de Contexto | [YouTube](https://youtu.be/F5zqRV7gEag) |
| 13 | Memoria de Agentes | [YouTube](https://youtu.be/QrYbHesIxpw) |
| 14 | Marco de Agentes de Microsoft | - |

## Technical Stack Covered

- Microsoft Agent Framework (MAF)
- Azure AI Agent Service
- Semantic Kernel
- AutoGen

## Image Storage

Images are stored in yoDEV R2 bucket:
```
https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/ai-agents-es/images/
```

See `image-mapping.json` for complete URL mapping.

## Processing Notes

1. Content compiled from Microsoft's Spanish translation (GitHub Actions machine-translated)
2. Images downloaded and re-hosted on R2
3. For Workplace import: base64 embedding used for reliable rendering
4. Videos embedded using Workplace's YouTube embed feature
5. Native Workplace TOC used instead of markdown TOC (anchor links don't work)

## Files (Not in Repo)

The following large files are generated locally but not committed:
- `ai-agents-course-es.md` - Original compiled markdown (~50KB)
- `ai-agents-course-es-final-v2.md` - Final with base64 images (~7.5MB)
- `images/` - Downloaded image files (~5MB total)

## Regenerating the Course

```bash
# 1. Download images
./scripts/download-images.sh

# 2. Add video links
python scripts/add-videos.py

# 3. Remove TOC
sed '/^## Tabla de Contenidos$/,/^---$/d' input.md > output.md

# 4. Convert images to base64
python scripts/convert-to-base64.py

# 5. Import to Workplace via markdown import function
```
