"""Render the submitted Markdown answers as a clean, two-page PDF."""
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

root = Path(__file__).resolve().parents[1]
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyAnswer', fontName='Helvetica', fontSize=10.5, leading=15, spaceAfter=9))
styles.add(ParagraphStyle(name='FormulaAnswer', fontName='Courier', fontSize=9, leading=13, leftIndent=12, spaceAfter=9))
output = root / 'submission/theory_answers.pdf'
doc = SimpleDocTemplate(str(output), pagesize=(612, 792), rightMargin=48, leftMargin=48, topMargin=44, bottomMargin=44)
story = []
source = (root / 'submission/theory_answers.md').read_text()
for block in source.split('\n\n'):
    block = block.strip('\n')
    if not block:
        continue
    if block.startswith('# '):
        story.append(Paragraph('Assignment 2', styles['Title']))
        story.append(Paragraph('Theory: Arithmetic of CNNs', styles['Heading2']))
    elif block.startswith('## '):
        if block.startswith('## Question 5'):
            story.append(PageBreak())
        story.append(Paragraph(escape(block[3:]), styles['Heading2']))
    elif block.startswith('    '):
        text = '<br/>'.join(escape(line.strip()).replace('×', 'x').replace('−', '-') for line in block.splitlines())
        story.append(Paragraph(text, styles['FormulaAnswer']))
    else:
        import re
        text = escape(block).replace('\n', ' ')
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        story.append(Paragraph(text, styles['BodyAnswer']))
def footer(canvas, doc):
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(564, 26, f'Assignment 2 | {doc.page}')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(output)
