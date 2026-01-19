#!/usr/bin/env python3
"""
Compile HF NLP Course Spanish MDX files into a single markdown document
suitable for Workplace import.
"""

import os
import re
from pathlib import Path

# Course structure - chapters 0-8 (Part 1)
COURSE_STRUCTURE = [
    {
        "chapter": 0,
        "title": "0. Configuración",
        "sections": [
            (1, "Introducción"),
        ]
    },
    {
        "chapter": 1,
        "title": "1. Modelos de Transformadores",
        "sections": [
            (1, "Introducción"),
            (2, "Procesamiento de Lenguaje Natural y LLMs"),
            (3, "Transformadores, ¿qué pueden hacer?"),
            (4, "¿Cómo funcionan los Transformadores?"),
            (5, "Cómo 🤗 Transformers resuelve tareas"),
            (6, "Arquitecturas de Transformadores"),
            (7, "Quiz rápido"),
            (8, "Inferencia con LLMs"),
            (9, "Sesgos y limitaciones"),
            (10, "Resumen"),
            (11, "Examen de certificación"),
        ]
    },
    {
        "chapter": 2,
        "title": "2. Usando 🤗 Transformers",
        "sections": [
            (1, "Introducción"),
            (2, "Detrás del pipeline"),
            (3, "Modelos"),
            (4, "Tokenizadores"),
            (5, "Manejando secuencias múltiples"),
            (6, "Poniendo todo junto"),
            (7, "¡Uso básico completado!"),
            (8, "Despliegue de Inferencia Optimizada"),
            (9, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 3,
        "title": "3. Ajuste fino de un modelo preentrenado",
        "sections": [
            (1, "Introducción"),
            (2, "Procesamiento de los datos"),
            (3, "Ajuste de un modelo con la API Trainer"),
            ("3_tf", "Ajuste de un modelo con Keras"),
            (4, "Un bucle de entrenamiento completo"),
            (5, "Ajuste fino, ¡hecho!"),
            (6, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 4,
        "title": "4. Compartiendo modelos y tokenizadores",
        "sections": [
            (1, "El Hub de Hugging Face"),
            (2, "Usando modelos preentrenados"),
            (3, "Compartiendo modelos preentrenados"),
            (4, "Construyendo una tarjeta de modelo"),
            (5, "¡Parte 1 completada!"),
            (6, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 5,
        "title": "5. La librería 🤗 Datasets",
        "sections": [
            (1, "Introducción"),
            (2, "¿Y si mi dataset no está en el Hub?"),
            (3, "Es momento de subdividir"),
            (4, "¿Big data? 🤗 ¡Datasets al rescate!"),
            (5, "Crea tu propio dataset"),
            (6, "Búsqueda semántica con FAISS"),
            (7, "🤗 Datasets, ¡listo!"),
            (8, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 6,
        "title": "6. La librería 🤗 Tokenizers",
        "sections": [
            (1, "Introducción"),
            (2, "Entrenar un nuevo tokenizador a partir de uno existente"),
            (3, "Los poderes especiales de los Tokenizadores Rápidos"),
            ("3b", "Tokenizadores Rápidos en el pipeline de QA"),
            (4, "Normalización y pre-tokenización"),
            (5, "Tokenización por Codificación Byte-Pair"),
            (6, "Tokenización WordPiece"),
            (7, "Tokenización Unigram"),
            (8, "Construir un tokenizador, bloque por bloque"),
            (9, "Tokenizadores, ¡listo!"),
            (10, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 7,
        "title": "7. Tareas clásicas de NLP",
        "sections": [
            (1, "Introducción"),
            (2, "Clasificación de tokens"),
            (3, "Ajuste fino de un modelo de lenguaje enmascarado"),
            (4, "Traducción"),
            (5, "Resúmenes"),
            (6, "Entrenamiento de un modelo de lenguaje causal desde cero"),
            (7, "Respuesta a preguntas"),
            (8, "Dominando los LLMs"),
            (9, "Quiz de fin de capítulo"),
        ]
    },
    {
        "chapter": 8,
        "title": "8. ¿Cómo solicitar ayuda?",
        "sections": [
            (1, "Introducción"),
            (2, "¿Qué hacer cuando se produce un error?"),
            (3, "Pidiendo ayuda en los foros"),
            (4, "Depurando el pipeline de entrenamiento"),
            ("4_tf", "Depurando el pipeline de entrenamiento (TensorFlow)"),
            (5, "Cómo escribir un buen issue"),
            (6, "¡Parte 2 completada!"),
            (7, "Quiz de fin de capítulo"),
        ]
    },
]

def clean_mdx_components(content: str) -> str:
    """Convert MDX components to markdown equivalents."""

    # Remove FrameworkSwitchCourse
    content = re.sub(r'<FrameworkSwitchCourse\s*\{fw\}\s*/>', '', content)

    # Remove CourseFloatingBanner
    content = re.sub(r'<CourseFloatingBanner[^>]*/?>', '', content)

    # Remove DISABLE-FRONTMATTER-SECTIONS comment
    content = re.sub(r'<!--\s*DISABLE-FRONTMATTER-SECTIONS\s*-->', '', content)

    # Convert Youtube component to embed link
    def youtube_replacer(match):
        video_id = match.group(1)
        return f'\n**Video:** [Ver en YouTube](https://youtu.be/{video_id})\n'
    content = re.sub(r'<Youtube\s+id=["\']([^"\']+)["\']\s*/?>', youtube_replacer, content)

    # Convert Tip/Warning/Note to blockquotes
    def tip_replacer(match):
        inner = match.group(1).strip()
        return f'\n> **💡 Tip:** {inner}\n'
    content = re.sub(r'<Tip>(.*?)</Tip>', tip_replacer, content, flags=re.DOTALL)

    # Remove DocNotebookDropdown
    content = re.sub(r'<DocNotebookDropdown[^>]*/?>', '', content)
    content = re.sub(r'<DocNotebookDropdown[^>]*>.*?</DocNotebookDropdown>', '', content, flags=re.DOTALL)

    # Handle framework conditionals - keep both versions with labels
    def framework_replacer(match):
        full_match = match.group(0)
        # Extract PT content
        pt_match = re.search(r'\{#if\s+fw\s*===\s*[\'"]pt[\'"]\}(.*?)(?:\{:else\}|\{/if\})', full_match, re.DOTALL)
        tf_match = re.search(r'\{:else\}(.*?)\{/if\}', full_match, re.DOTALL)

        result = ""
        if pt_match:
            pt_content = pt_match.group(1).strip()
            if pt_content:
                result += f"\n**PyTorch:**\n\n{pt_content}\n"
        if tf_match:
            tf_content = tf_match.group(1).strip()
            if tf_content:
                result += f"\n**TensorFlow/Keras:**\n\n{tf_content}\n"

        return result if result else ""

    content = re.sub(r'\{#if\s+fw\s*===\s*[\'"]pt[\'"]\}.*?\{/if\}', framework_replacer, content, flags=re.DOTALL)

    # Convert Question component to quiz format - simplified approach
    # Just keep the Question component structure as markdown list
    def question_replacer(match):
        inner = match.group(0)
        result = "\n"

        # Extract text values from choices
        texts = re.findall(r'text:\s*["\']([^"\']+)["\']', inner)
        for text in texts:
            # Clean up HTML code tags
            text = re.sub(r'<code>([^<]+)</code>', r'`\1`', text)
            result += f"- {text}\n"

        return result if texts else ""

    content = re.sub(r'<Question\s+choices=\{.*?\}\s*/>', question_replacer, content, flags=re.DOTALL)

    # Clean up extra whitespace
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    return content

def read_mdx_file(filepath: Path) -> str:
    """Read and clean an MDX file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return clean_mdx_components(content)
    except FileNotFoundError:
        return ""

def compile_course(source_dir: Path, output_file: Path):
    """Compile all MDX files into a single markdown document."""

    output_lines = [
        "# Curso de NLP de Hugging Face 🤗",
        "",
        "**Parte 1: Fundamentos de Transformadores y NLP**",
        "",
        "Este curso te enseñará los fundamentos del procesamiento de lenguaje natural (NLP) usando las bibliotecas del ecosistema Hugging Face: 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers y 🤗 Accelerate.",
        "",
        "---",
        "",
    ]

    for chapter_info in COURSE_STRUCTURE:
        chapter_num = chapter_info["chapter"]
        chapter_title = chapter_info["title"]

        print(f"Processing {chapter_title}...")
        output_lines.append(f"# {chapter_title}")
        output_lines.append("")

        for section in chapter_info["sections"]:
            section_num, section_title = section

            # Build file path
            filename = f"{section_num}.mdx"
            filepath = source_dir / f"chapter{chapter_num}" / filename

            content = read_mdx_file(filepath)
            if content:
                # Add section header if not already present
                if not content.strip().startswith("#"):
                    output_lines.append(f"## {section_title}")
                    output_lines.append("")

                output_lines.append(content)
                output_lines.append("")
                output_lines.append("---")
                output_lines.append("")
            else:
                print(f"  Warning: Missing {filepath}")

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\nCompiled course written to: {output_file}")
    print(f"Total size: {output_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    source_dir = Path("/Users/grego/hf-course/chapters/es")
    output_file = Path("/Users/grego/ai-agents-course-images/workplace-course-integration/courses/hf-nlp-es/hf-nlp-course-es.md")

    compile_course(source_dir, output_file)
