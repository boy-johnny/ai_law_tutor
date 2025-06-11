import re
import json
from pathlib import Path

def parse_law_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    law_name = ""
    current_chapter = ""
    current_section = ""
    articles = []
    article = None

    # 正則表達式
    law_name_re = re.compile(r"法規名稱：(.+)")
    chapter_re = re.compile(r"第\s*([一二三四五六七八九十百千]+)\s*章\s*(.*)")
    section_re = re.compile(r"第\s*([一二三四五六七八九十百千]+)\s*節\s*(.*)")
    article_re = re.compile(r"第\s*(\d+)\s*條")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 法規名稱
        m = law_name_re.match(line)
        if m:
            law_name = m.group(1)
            continue

        # 章
        m = chapter_re.match(line)
        if m:
            current_chapter = m.group(0)
            continue

        # 節
        m = section_re.match(line)
        if m:
            current_section = m.group(0)
            continue

        # 條文
        m = article_re.match(line)
        if m:
            if article:
                articles.append(article)
            article = {
                "law_name": law_name,
                "chapter": current_chapter,
                "section": current_section,
                "article_no": m.group(1),
                "content": ""
            }
            # 條文標題後可能有內容
            content = line[m.end():].strip()
            if content:
                article["content"] += content
            continue

        # 條文內容
        if article:
            if article["content"]:
                article["content"] += "\n"
            article["content"] += line

    # 最後一條
    if article:
        articles.append(article)

    return articles

def main():
    # 你可以把所有檔案路徑放在這個 list
    files = [
        "raw_data/law/訴願法.md",
        "raw_data/law/行政訴訟法.md",
        "raw_data/law/行政罰法.md",
        "raw_data/law/行政程序法.md",
        "raw_data/law/行政執行法.md"
    ]
    for file in files:
        articles = parse_law_markdown(file)
        json_path = Path(file).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"已轉換 {file} -> {json_path}")

if __name__ == "__main__":
    main()