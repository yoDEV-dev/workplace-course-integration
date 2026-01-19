#!/usr/bin/env python3
"""Add video links to course lessons"""

import re

# Video URLs by lesson number
VIDEOS = {
    "01": "https://youtu.be/3zgm60bXmQk",
    "02": "https://youtu.be/ODwF-EZo_O8",
    "03": "https://youtu.be/m9lM8qqoOEA",
    "04": "https://youtu.be/vieRiPRx-gI",
    "05": "https://youtu.be/WcjAARvdL7I",
    "06": "https://youtu.be/iZKkMEGBCUQ",
    "07": "https://youtu.be/kPfJ2BrBCMY",
    "08": "https://youtu.be/V6HpE9hZEx0",
    "09": "https://youtu.be/His9R6gw6Ec",
    "10": "https://youtu.be/l4TP6IyJxmQ",
    "11": "https://youtu.be/X-Dh9R3Opn8",
    "12": "https://youtu.be/F5zqRV7gEag",
    "13": "https://youtu.be/QrYbHesIxpw",
}

def add_video_links(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # For each lesson with a video, add the link after the thumbnail
    for lesson_num, video_url in VIDEOS.items():
        # Pattern: find the lesson heading and its thumbnail image
        # Example: ## Lección 01: ... \n\n![Lección 01](...)
        pattern = rf'(## Lección {lesson_num}: [^\n]+\n\n!\[Lección {lesson_num}\]\([^)]+\))'

        replacement = rf'\1\n\n**Video de la lección:** {video_url}'

        content = re.sub(pattern, replacement, content)
        print(f"Added video for Lección {lesson_num}")

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"\nOutput written to: {output_file}")

if __name__ == '__main__':
    add_video_links(
        'ai-agents-course-es.md',
        'ai-agents-course-es-with-videos.md'
    )
