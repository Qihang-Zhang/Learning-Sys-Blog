#!/usr/bin/env python3
import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MOTIVATION = "# Tutorial Towards Large-Scale Learning Systems \n\n The idea of writing a Learning Blog has been on my mind for a long time. I have always been inspired by [Lil's Log](https://lilianweng.github.io/). Recently, Chenyang's work on [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial?tab=readme-ov-file) further inspired me. \n\n Additionally, as I introduced [here](docs/Tutorial/motivation-en.md), I feel that the research focus and skill set required for Large-Scale Learning Systems are evolving rapidly. Given all these influences, I decided to start my own blog to organize my learnings, thoughts, and reflections, while also sharing insights and sparking discussions with others. \n\n"

class Post:
    def __init__(self, filepath: Path, date: datetime, title: str, lang: str, emoji: str = "📌"):
        self.filepath = filepath
        self.date = date
        self.title = title
        self.lang = lang
        self.emoji = emoji
        self.relative_path = str(filepath).replace("\\", "/")
        self.slug = None  # Will be set from front matter if available
        
        # Extract slug from front matter if available
        self._extract_slug_from_front_matter()
        
    def _extract_slug_from_front_matter(self):
        """Extract slug from front matter."""
        try:
            content = self.filepath.read_text(encoding='utf-8')
            if content.startswith('---'):
                end = content.find('---', 3)
                if end != -1:
                    front_matter = yaml.safe_load(content[3:end])
                    if front_matter and 'slug' in front_matter:
                        self.slug = front_matter['slug']
        except Exception:
            pass
        
    def get_mkdocs_url(self) -> str:
        """Generate mkdocs-compatible URL for the post."""
        date_str = self.date.strftime("%Y/%m/%d")
        # Use slug from front matter if available, otherwise use title
        slug = self.slug if self.slug else self.title
        # Replace spaces with %20 for URL encoding
        slug = slug.replace(" ", "%20")
        return f"https://qihang-zhang.github.io/Large-Scale-Learning-Sys-Tutorial/Tutorial/{date_str}/{slug}.html"

class ReadmeUpdater:
    def __init__(self, posts_dir: str = "docs/Tutorial/Posts", readme_file: str = "README.md"):
        self.posts_dir = PROJECT_ROOT / posts_dir
        self.readme_file = PROJECT_ROOT / readme_file
        self.posts: Dict[str, Dict[str, Post]] = {}  # key: base_name, value: {"en": Post, "zh": Post}
        self.emoji_map = ["📌", "✍️", "📖", "🔍", "👋"]
        self.emoji_index = 0
        
    def get_next_emoji(self) -> str:
        emoji = self.emoji_map[self.emoji_index % len(self.emoji_map)]
        self.emoji_index += 1
        return emoji

    def extract_front_matter(self, file_path: Path) -> Optional[dict]:
        content = file_path.read_text(encoding='utf-8')
        if content.startswith('---'):
            try:
                end = content.find('---', 3)
                if end != -1:
                    front_matter = yaml.safe_load(content[3:end])
                    return front_matter
            except yaml.YAMLError:
                pass
        return None

    def parse_filename(self, file_path: Path) -> Optional[Tuple[datetime, str, str]]:
        pattern = r"(.+)-(en|zh)\.md$"
        match = re.match(pattern, file_path.name)
        if not match:
            return None
        
        title, lang = match.groups()
        title = title.replace("-", " ")
        date = datetime.now()
        return date, title, lang

    def scan_posts(self):
        self.posts.clear()
        if not self.posts_dir.exists():
            return
            
        for file_path in self.posts_dir.glob("*.md"):
            parsed = self.parse_filename(file_path)
            if not parsed:
                continue

            date, title, lang = parsed
            
            # 检查 Front Matter
            front_matter = self.extract_front_matter(file_path)
            if front_matter:
                # 日期是必需的
                if 'date' in front_matter:
                    try:
                        date = datetime.strptime(str(front_matter['date']), "%Y-%m-%d")
                    except ValueError:
                        print(f"Warning: Invalid date format in {file_path}, skipping...")
                        continue
                else:
                    print(f"Warning: No date found in front matter of {file_path}, skipping...")
                    continue
                
                if 'slug' in front_matter:
                    title = front_matter['slug']

            base_name = file_path.name[:-6]  
            
            if base_name not in self.posts:
                self.posts[base_name] = {}
            
            emoji = self.get_next_emoji()
            self.posts[base_name][lang] = Post(file_path, date, title, lang, emoji)
        
    def generate_post_list(self) -> str:
        if not self.posts:
            return "No posts found."

        categories: Dict[str, List[Tuple[str, Dict[str, Post]]]] = {}
        
        for base_name, post_versions in self.posts.items():
            categories_list = []
            if "en" in post_versions and (front_matter := self.extract_front_matter(post_versions["en"].filepath)):
                categories_list = front_matter.get("categories", ["Uncategorized"])
            elif "zh" in post_versions and (front_matter := self.extract_front_matter(post_versions["zh"].filepath)):
                categories_list = front_matter.get("categories", ["Uncategorized"])
            else:
                categories_list = ["Uncategorized"]

            for category in categories_list:
                if category not in categories:
                    categories[category] = []
                categories[category].append((base_name, post_versions))

        lines = ["# Latest Articles\n"]
        
        sorted_categories = sorted(categories.keys())
        
        for category in sorted_categories:
            lines.append(f"\n## {category}")
            
            sorted_posts = sorted(
                categories[category],
                key=lambda x: max(
                    post.date for post in x[1].values()
                ),
                reverse=True
            )
            
            for _, post_versions in sorted_posts:
                date = max(post.date for post in post_versions.values())
                date_str = date.strftime("%Y-%m-%d")
                
                title = None
                if "en" in post_versions:
                    title = post_versions["en"].title
                elif "zh" in post_versions:
                    title = post_versions["zh"].title
                
                links = []
                if "en" in post_versions:
                    en_url = post_versions["en"].get_mkdocs_url()
                    links.append(f'[English Version]({en_url})')
                if "zh" in post_versions:
                    zh_url = post_versions["zh"].get_mkdocs_url()
                    links.append(f'[中文版]({zh_url})')
                
                line = f'- 📌 **{title}** ({" / ".join(links)}) ({date_str})'
                lines.append(line)
        
        return "\n".join(lines)

    def update_readme(self) -> bool:
        self.scan_posts()
        posts_content = self.generate_post_list()
        self.readme_file.write_text(MOTIVATION + posts_content, encoding='utf-8')
        print("README.md has been updated successfully.")
        return True


def main():
    updater = ReadmeUpdater()
    updater.update_readme()
    
if __name__ == "__main__":
    main()
