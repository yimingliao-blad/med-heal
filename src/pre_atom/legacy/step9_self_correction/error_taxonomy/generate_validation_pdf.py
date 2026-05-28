#!/usr/bin/env python3
"""Generate a PDF validation document for human review of GPT-4o error
classifications across 5 models.

The PDF contains:
- Instructions and taxonomy reference
- 50 items (10 per model), each showing:
  - Model name, Patient ID, Question
  - Ground truth answer
  - Model's (incorrect) answer
  - GPT-4o's classification + reasoning
  - A checkbox for the reviewer to agree/disagree + space for correction

Output: output/step9_v2/validation_review_50items.pdf
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "output" / "step9_v2" / "validation_sample_50.json"
NOTES_PATH = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
OUT_PDF = PROJECT_ROOT / "output" / "step9_v2" / "validation_review_50items.pdf"

MODEL_LABELS = {
    "biomistral-7b": "BioMistral-7B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "qwen3-8b": "Qwen3-8B",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-Distill-8B",
}

TAXONOMY_TEXT = """
<b>MISREADING</b> — The model misread, misinterpreted, or confused information that IS in the
discharge notes. The source material exists but was understood incorrectly
(e.g., wrong dosage, confused two medications, mixed up two visits).<br/><br/>

<b>FABRICATION</b> — The model states something that is NOT in the discharge notes at all.
It invented or hallucinated a clinical detail (medication, procedure, diagnosis, date)
with no basis in the notes.<br/><br/>

<b>OMISSION</b> — The model failed to mention critical information that IS in the notes and
is needed to answer the question correctly. The answer is incomplete in a way that
changes the conclusion.<br/><br/>

<b>QUESTION_MISALIGNMENT</b> — The model answered a different question than what was asked,
or focused on irrelevant aspects of the notes while missing the actual question's focus
(e.g., wrong visit, wrong time period, wrong clinical aspect).<br/><br/>

<b>HEDGING</b> — The model provides multiple possible answers or hedges instead of committing
to what the notes clearly state. The correct information may be mentioned but is diluted
by alternatives.<br/><br/>

<b>OTHER</b> — None of the above categories fit.
"""


def load_notes() -> dict[int, str]:
    df = pd.read_json(NOTES_PATH, lines=True)
    out = {}
    for _, r in df.iterrows():
        pid = int(r["patient_id"])
        parts = []
        for i in (1, 2, 3):
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(t)
        out[pid] = "\n---\n".join(parts)
    return out


def truncate(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, {len(text)} chars total]"


def sanitize(text: str) -> str:
    """Make text safe for ReportLab XML parser."""
    if not text:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Replace non-breaking spaces and other control chars
    text = text.replace("\x00", "")
    return text


def build_pdf():
    sample = json.load(open(SAMPLE_PATH))
    notes = load_notes()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16,
                                  spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13,
                                    spaceAfter=6, spaceBefore=12)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9,
                                 leading=12, spaceAfter=4)
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8,
                                  leading=10, spaceAfter=2, textColor=colors.HexColor("#333333"))
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=7.5,
                                 leading=9.5, spaceAfter=2,
                                 textColor=colors.HexColor("#444444"),
                                 backColor=colors.HexColor("#F5F5F5"))
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9,
                                  leading=11, textColor=colors.HexColor("#1a5276"),
                                  spaceAfter=2)
    checkbox_style = ParagraphStyle("Checkbox", parent=styles["Normal"], fontSize=10,
                                     leading=14, spaceBefore=6, spaceAfter=2)
    taxonomy_style = ParagraphStyle("Taxonomy", parent=styles["Normal"], fontSize=9,
                                     leading=12, spaceAfter=4)

    story = []

    # --- Cover page ---
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("Hallucination Error Classification<br/>Validation Review",
                            title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This document contains <b>50 items</b> (10 per model) where GPT-4o has classified "
        "the type of hallucination error in an LLM's incorrect answer to a clinical question "
        "from the EHRNoteQA benchmark (MIMIC-IV discharge summaries).<br/><br/>"
        "For each item, please:<br/>"
        "1. Read the question, ground truth, and model's answer<br/>"
        "2. Review the discharge note excerpt for context<br/>"
        "3. Read GPT-4o's error classification and reasoning<br/>"
        "4. <b>Mark whether you AGREE or DISAGREE</b> with the classification<br/>"
        "5. If you disagree, write the correct category<br/><br/>"
        "Your review will take approximately <b>2–3 hours</b>.",
        body_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<b>Reviewer Name:</b> ___________________________", body_style))
    story.append(Paragraph("<b>Date:</b> ___________________________", body_style))
    story.append(PageBreak())

    # --- Taxonomy reference page ---
    story.append(Paragraph("Error Type Taxonomy Reference", heading_style))
    story.append(Paragraph(TAXONOMY_TEXT, taxonomy_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "<b>Note:</b> Each item has exactly ONE primary error type. If multiple errors are "
        "present, select the one that is the <i>primary cause</i> of the answer being incorrect. "
        "For example, if the model both fabricated a fact AND omitted another, choose the one "
        "that is more responsible for the wrong answer.",
        small_style))
    story.append(PageBreak())

    # --- Items ---
    for idx, item in enumerate(sample, 1):
        model = MODEL_LABELS.get(item.get("model", ""), item.get("model", "?"))
        pid = item.get("patient_id", "?")
        question = item.get("question", "")
        gt = item.get("ground_truth", "")
        answer = item.get("model_answer", "")
        error_type = item.get("primary_error", item.get("PRIMARY_ERROR", "?"))
        description = item.get("description", item.get("ERROR_DESCRIPTION", ""))
        severity = item.get("severity", item.get("SEVERITY", "?"))

        note_text = notes.get(int(pid), "")

        # Header
        story.append(Paragraph(
            f"Item {idx} / 50 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Model:</b> {sanitize(model)} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Patient ID:</b> {pid}",
            heading_style))

        # Question
        story.append(Paragraph("<b>Question:</b>", label_style))
        story.append(Paragraph(sanitize(str(question)[:500]), body_style))

        # Ground truth
        story.append(Paragraph("<b>Ground Truth (Correct Answer):</b>", label_style))
        story.append(Paragraph(sanitize(str(gt)[:500]), body_style))

        # Model answer
        story.append(Paragraph(f"<b>{sanitize(model)}'s Answer (Incorrect):</b>", label_style))
        story.append(Paragraph(sanitize(truncate(str(answer), 800)), body_style))

        # Discharge note (truncated)
        story.append(Paragraph("<b>Discharge Note Excerpt:</b>", label_style))
        story.append(Paragraph(sanitize(truncate(note_text, 1500)), note_style))

        # GPT-4o classification box
        story.append(Spacer(1, 0.15 * inch))
        box_data = [
            ["GPT-4o Classification", ""],
            ["Error Type:", f"{error_type}"],
            ["Severity:", f"{severity}"],
            ["Reasoning:", sanitize(truncate(str(description), 400))],
        ]
        # Wrap long text in Paragraphs
        box_table_data = []
        for row in box_data:
            left = Paragraph(f"<b>{sanitize(row[0])}</b>", small_style)
            right = Paragraph(sanitize(str(row[1])), small_style)
            box_table_data.append([left, right])

        box = Table(box_table_data, colWidths=[1.3 * inch, 5.2 * inch])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5E8D4")),
            ("SPAN", (0, 0), (1, 0)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#82B366")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(box)

        # Reviewer response
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(
            "\u2610 <b>AGREE</b> with GPT-4o's classification &nbsp;&nbsp;&nbsp;&nbsp;"
            "\u2610 <b>DISAGREE</b> — Correct category: _______________",
            checkbox_style))
        story.append(Paragraph(
            "Comments: _______________________________________________________________",
            checkbox_style))

        # Page break after each item
        story.append(PageBreak())

    # --- Summary page ---
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(
        "Thank you for completing this review. Please record your totals below:",
        body_style))
    story.append(Spacer(1, 0.2 * inch))

    summary_data = [
        ["", "Count"],
        ["Total items reviewed", "/ 50"],
        ["AGREE with GPT-4o", ""],
        ["DISAGREE with GPT-4o", ""],
        ["", ""],
        ["Disagreements by my corrected type:", ""],
        ["  → MISREADING", ""],
        ["  → FABRICATION", ""],
        ["  → OMISSION", ""],
        ["  → QUESTION_MISALIGNMENT", ""],
        ["  → HEDGING", ""],
        ["  → OTHER", ""],
    ]
    summary_table = Table(summary_data, colWidths=[3.5 * inch, 1.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<b>Reviewer Signature:</b> ___________________________ "
        "&nbsp;&nbsp;&nbsp; <b>Date:</b> _______________",
        body_style))

    # Build PDF
    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    doc.build(story)
    print(f"Wrote {OUT_PDF} ({len(sample)} items)")


if __name__ == "__main__":
    build_pdf()
