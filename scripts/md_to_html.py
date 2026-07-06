#!/usr/bin/env python3
"""Minimal Markdown -> HTML converter for FINAL_PROJECT_BOOK.md.

Handles the constructs used in the book: ATX headings, GFM pipe tables,
fenced code blocks, unordered/ordered lists, horizontal rules, blockquotes,
and inline bold/italic/code. Produces styled HTML that macOS `textutil` can
convert to .doc/.docx/.rtf with tables intact. Standard-library only.

Usage: python3 md_to_html.py INPUT.md OUTPUT.html
"""
from __future__ import annotations
import html
import re
import sys


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def is_table_sep(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", line))


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    in_code = False
    while i < n:
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                out.append("<pre><code>")
            else:
                in_code = False
                out.append("</code></pre>")
            i += 1
            continue
        if in_code:
            out.append(html.escape(line, quote=False))
            i += 1
            continue

        if re.match(r"^---+\s*$", line):
            out.append("<hr/>")
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue

        if "|" in line and i + 1 < n and is_table_sep(lines[i + 1]):
            header = split_row(line)
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(split_row(lines[i]))
                i += 1
            out.append('<table border="1" cellspacing="0" cellpadding="4">')
            out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead>")
            out.append("<tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue

        if re.match(r"^\s*-\s+", line):
            out.append("<ul>")
            while i < n and re.match(r"^\s*-\s+", lines[i]):
                item = re.sub(r"^\s*-\s+", "", lines[i])
                out.append(f"<li>{inline(item)}</li>")
                i += 1
            out.append("</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            out.append("<ol>")
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                out.append(f"<li>{inline(item)}</li>")
                i += 1
            out.append("</ol>")
            continue

        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        if not line.strip():
            i += 1
            continue

        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|---+\s*$|\s*[-\d]|>|```)", lines[i]) and "|" not in lines[i]:
            buf.append(lines[i])
            i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")

    body = "\n".join(out)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.4; }}
h1 {{ font-size: 20pt; }} h2 {{ font-size: 16pt; }} h3 {{ font-size: 13pt; }}
h4 {{ font-size: 12pt; }}
table {{ border-collapse: collapse; margin: 8pt 0; font-size: 10pt; }}
th, td {{ border: 1px solid #666; padding: 3pt 6pt; text-align: left; }}
th {{ background: #eee; }}
code {{ font-family: 'Courier New', monospace; font-size: 10pt; }}
pre {{ background: #f4f4f4; padding: 6pt; font-size: 9pt; }}
blockquote {{ margin-left: 12pt; color: #444; border-left: 3px solid #ccc; padding-left: 8pt; }}
</style></head><body>
{body}
</body></html>"""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        md = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(convert(md))
    print(f"wrote {dst}")
