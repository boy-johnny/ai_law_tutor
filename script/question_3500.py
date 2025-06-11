import re
import json

def parse_question_line(line):
    # 正則：抓取 (答案)題號 題目內容(A)選項A...(B)選項B...(C)選項C...(D)選項D...
    pattern = r'^\((?P<answer>[A-D])\)(?P<number>\d+)\s*(?P<question>.*?)(?=\(A\))\(A\)(?P<A>.*?)(?=\(B\))\(B\)(?P<B>.*?)(?=\(C\))\(C\)(?P<C>.*?)(?=\(D\))\(D\)(?P<D>.*)$'
    match = re.match(pattern, line)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "number": groups["number"],
        "question": groups["question"].strip(),
        "options": {
            "A": groups["A"].strip(),
            "B": groups["B"].strip(),
            "C": groups["C"].strip(),
            "D": groups["D"].strip()
        },
        "answer": groups["answer"]
    }

def parse_questions_file(filepath):
    questions = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('('):
                continue
            q = parse_question_line(line)
            if q:
                questions.append(q)
    return questions

def main():
    input_file = "raw_data/textbook/行政法3500題.md"
    output_file = "raw_data/textbook/行政法3500題.json"
    questions = parse_questions_file(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"已轉換 {len(questions)} 題，輸出到 {output_file}")

if __name__ == "__main__":
    main()