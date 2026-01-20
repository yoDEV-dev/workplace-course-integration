#!/usr/bin/env python3
"""
Compile ML-For-Beginners Spanish course into markdown documents
with base64 embedded images, split into 4 parts.
"""

import os
import re
import base64
from pathlib import Path

# Source paths
REPO_ROOT = Path("/Users/grego/ml-for-beginners")
ES_ROOT = REPO_ROOT / "translations" / "es"
OUTPUT_DIR = Path("/Users/grego/ai-agents-course-images/workplace-course-integration/courses/ml-beginners-es/source")

# Course structure for 4 parts
COURSE_PARTS = {
    "part1-fundamentos": {
        "title": "Parte 1: Fundamentos",
        "subtitle": "Introducción al Machine Learning y Regresión",
        "sections": [
            ("1-Introduction", "Introducción al Machine Learning"),
            ("2-Regression", "Regresión"),
        ]
    },
    "part2-clasificacion": {
        "title": "Parte 2: Clasificación",
        "subtitle": "Aplicaciones Web, Clasificación y Clustering",
        "sections": [
            ("3-Web-App", "Aplicaciones Web"),
            ("4-Classification", "Clasificación"),
            ("5-Clustering", "Clustering"),
        ]
    },
    "part3-nlp": {
        "title": "Parte 3: Procesamiento de Lenguaje Natural",
        "subtitle": "NLP y Análisis de Texto",
        "sections": [
            ("6-NLP", "Procesamiento de Lenguaje Natural"),
        ]
    },
    "part4-avanzado": {
        "title": "Parte 4: Temas Avanzados",
        "subtitle": "Series de Tiempo, Aprendizaje por Refuerzo y Aplicaciones",
        "sections": [
            ("7-TimeSeries", "Series de Tiempo"),
            ("8-Reinforcement", "Aprendizaje por Refuerzo"),
            ("9-Real-World", "Aplicaciones del Mundo Real"),
        ]
    }
}

def get_mime_type(filepath: str) -> str:
    """Get MIME type from file extension."""
    ext = filepath.lower().split('.')[-1]
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/png')

def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 data URI."""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        mime = get_mime_type(str(image_path))
        return f"data:{mime};base64,{data}"
    except Exception as e:
        print(f"  Warning: Could not read image {image_path}: {e}")
        return str(image_path)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main"

def resolve_to_github_url(md_file: Path, relative_path: str) -> str:
    """Convert relative image path to GitHub raw URL."""
    # Handle paths like ../../../../1-Introduction/1-intro-to-ML/images/hype.png
    md_dir = md_file.parent
    resolved = (md_dir / relative_path).resolve()

    # Get path relative to repo root
    try:
        rel_to_repo = resolved.relative_to(REPO_ROOT)
        return f"{GITHUB_RAW_BASE}/{rel_to_repo}"
    except ValueError:
        # Path is outside repo, try to construct from the relative path
        clean_path = relative_path
        while clean_path.startswith('../'):
            clean_path = clean_path[3:]
        clean_path = clean_path.lstrip('./')
        return f"{GITHUB_RAW_BASE}/{clean_path}"

def process_markdown(md_file: Path) -> str:
    """Read markdown file and convert image paths to GitHub URLs."""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {md_file}: {e}")
        return ""

    # Find all image references: ![alt](path)
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        # Skip external URLs (http/https)
        if img_path.startswith('http://') or img_path.startswith('https://'):
            return match.group(0)

        # Convert to GitHub raw URL
        github_url = resolve_to_github_url(md_file, img_path)
        return f'![{alt_text}]({github_url})'

    content = re.sub(img_pattern, replace_image, content)
    return content

def get_lessons_in_section(section_dir: Path) -> list:
    """Get ordered list of lesson directories in a section."""
    lessons = []
    if not section_dir.exists():
        return lessons

    for item in sorted(section_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Skip solution/Julia/R folders
            if item.name in ['solution', 'data', 'quiz-app', 'docs', 'sketchnotes']:
                continue
            lessons.append(item)

    return lessons

def compile_section(section_name: str, section_title: str) -> str:
    """Compile all lessons in a section."""
    output = []
    section_dir = ES_ROOT / section_name

    if not section_dir.exists():
        print(f"  Warning: Section directory not found: {section_dir}")
        return ""

    # Add section header
    output.append(f"\n# {section_title}\n")

    # Add section README if exists
    section_readme = section_dir / "README.md"
    if section_readme.exists():
        content = process_markdown(section_readme)
        # Remove the first H1 if it exists (we already added section title)
        content = re.sub(r'^# .+\n', '', content, count=1)
        output.append(content)
        output.append("\n---\n")

    # Process each lesson
    lessons = get_lessons_in_section(section_dir)
    for lesson_dir in lessons:
        readme = lesson_dir / "README.md"
        if readme.exists():
            print(f"  Processing: {lesson_dir.name}")
            content = process_markdown(readme)
            output.append(content)
            output.append("\n---\n")

    return '\n'.join(output)

def compile_part(part_key: str, part_info: dict) -> str:
    """Compile a complete part of the course."""
    output = []

    # Add part header
    output.append(f"# ML para Principiantes 🤖")
    output.append(f"## {part_info['title']}")
    output.append(f"**{part_info['subtitle']}**")
    output.append("")
    output.append("---")
    output.append("")

    # Compile each section
    for section_name, section_title in part_info['sections']:
        print(f"Processing section: {section_name}")
        section_content = compile_section(section_name, section_title)
        output.append(section_content)

    return '\n'.join(output)

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling ML-For-Beginners Course (Spanish)")
    print("=" * 50)

    # Compile each part
    for part_key, part_info in COURSE_PARTS.items():
        print(f"\n{part_info['title']}")
        print("-" * 40)

        content = compile_part(part_key, part_info)

        output_file = OUTPUT_DIR / f"ml-beginners-{part_key}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        size_kb = output_file.stat().st_size / 1024
        print(f"  Written: {output_file.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 50)
    print("Compilation complete!")

if __name__ == "__main__":
    main()
