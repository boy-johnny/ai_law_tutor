import re
import json

def parse_md_to_rag_blocks(md_path, output_path):
    with open(md_path, encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    chapter = ""
    for line in lines:
        line = line.strip()
        # 章節
        if re.match(r'^\d+ .+', line):
            chapter = line
            continue
        # 選擇題
        m = re.match(r'（[Ａ-Ｄ]）▲ (.+?)（[Ａ-Ｄ]）(.+?)（[Ａ-Ｄ]）(.+?)（[Ａ-Ｄ]）(.+?)（[Ａ-Ｄ]）(.+?)。.*【(\d+ .+)】', line)
        if m:
            question = m.group(1)
            options = [m.group(2), m.group(3), m.group(4), m.group(5)]
            answer = m.group(6)
            year = m.group(7)
            blocks.append({
                "type": "qa",
                "question": question,
                "options": options,
                "answer": answer,
                "chapter": chapter,
                "source": md_path,
                "year": year
            })
            continue
        # 知識點
        if line and not line.startswith("解析") and not line.startswith("") and not line.startswith("【"):
            blocks.append({
                "type": "fact",
                "content": line,
                "chapter": chapter,
                "source": md_path
            })
        # 解析
        if line.startswith("解析") or line.startswith("") or line.startswith(""):
            blocks.append({
                "type": "explain",
                "content": line,
                "chapter": chapter,
                "source": md_path
            })

    # 輸出 JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

parse_md_to_rag_blocks("raw_data/textbook/行政法概論.md", "raw_data/textbook/行政法概論_rag.jsonl")