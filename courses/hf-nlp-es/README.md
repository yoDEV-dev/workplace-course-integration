# Curso de NLP de Hugging Face (Español)

## Open in Workplace

[**Abrir en Workplace**](https://workplace.yodev.dev/template/import?name=Curso%20de%20NLP%20de%20Hugging%20Face&mode=page&snapshotUrl=https://pub-f1a17e3d5e8149a18f5a54f2ea99c18c.r2.dev/courses/hf-nlp-es/course.zip)

## Información del Curso

- **Fuente Original:** [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- **Idioma:** Español
- **Capítulos:** 0-8 (Parte 1: Fundamentos)
- **Archivos MDX:** 70
- **Tamaño compilado:** ~920 KB

## Resumen de Contenido

| Capítulo | Título | Secciones |
|----------|--------|-----------|
| 0 | Configuración | 1 |
| 1 | Modelos de Transformadores | 11 |
| 2 | Usando 🤗 Transformers | 9 |
| 3 | Ajuste fino de un modelo preentrenado | 7 (incl. TF) |
| 4 | Compartiendo modelos y tokenizadores | 6 |
| 5 | La librería 🤗 Datasets | 8 |
| 6 | La librería 🤗 Tokenizers | 11 (incl. 3b) |
| 7 | Tareas clásicas de NLP | 9 |
| 8 | ¿Cómo solicitar ayuda? | 8 (incl. TF) |

## Temas Técnicos Cubiertos

- Arquitecturas de Transformadores (Encoder, Decoder, Seq2Seq)
- Pipeline de 🤗 Transformers
- Tokenización (BPE, WordPiece, Unigram)
- Ajuste fino con Trainer API y Keras
- Procesamiento de datasets con 🤗 Datasets
- Clasificación de tokens (NER)
- Modelado de lenguaje (MLM, CLM)
- Traducción y resúmenes
- Respuesta a preguntas
- Depuración de entrenamiento

## Notas de Procesamiento

1. Contenido traducido del inglés al español
2. Componentes MDX convertidos a markdown estándar:
   - `<Youtube>` → Enlaces a videos
   - `<Question>` → Listas de opciones
   - `<FrameworkSwitchCourse>` → Secciones etiquetadas PyTorch/TensorFlow
3. Código preservado sin cambios (solo comentarios traducidos)
4. Se mantienen términos técnicos comunes en la comunidad ML hispana

## Archivos

- `course.zip` - BlockSuite snapshot para importar en Workplace (~494KB)
- `source/hf-nlp-course-es.md` - Markdown fuente compilado (~920KB)
- `README.md` - Este archivo

## Regenerar el Curso

```bash
# Desde el directorio workplace-course-integration
python3 scripts/compile-hf-course.py
```

## Fuente de las Traducciones

Los archivos MDX en español se encuentran en:
```
/Users/grego/hf-course/chapters/es/
```

Estructura:
- `chapter0/` - 1 archivo
- `chapter1/` - 11 archivos
- `chapter2/` - 9 archivos
- `chapter3/` - 7 archivos (incluye 3_tf.mdx)
- `chapter4/` - 6 archivos
- `chapter5/` - 8 archivos
- `chapter6/` - 11 archivos (incluye 3b.mdx)
- `chapter7/` - 9 archivos
- `chapter8/` - 8 archivos (incluye 4_tf.mdx)
