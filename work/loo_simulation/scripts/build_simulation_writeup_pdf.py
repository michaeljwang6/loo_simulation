"""Render SIMULATION_WRITEUP.md as a publication-style PDF."""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "SIMULATION_WRITEUP.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "loo_simulation_writeup.pdf"

BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#D55E00")
DARK = colors.HexColor("#1F2937")
MID = colors.HexColor("#4B5563")
LIGHT = colors.HexColor("#E5E7EB")
VERY_LIGHT = colors.HexColor("#F8FAFC")
MATH_DPI = 300


class MathRenderer:
    """Render Markdown LaTeX fragments with Matplotlib's mathtext engine."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _render(self, expression: str, font_size: float) -> Path:
        normalized = " ".join(expression.strip().split())
        digest = hashlib.sha256(
            f"{font_size:.2f}|{normalized}".encode("utf-8")
        ).hexdigest()[:20]
        path = self.directory / f"math-{digest}.png"
        if not path.exists():
            math_to_image(
                f"${normalized}$",
                path,
                prop=FontProperties(
                    family="DejaVu Serif",
                    size=font_size,
                ),
                dpi=MATH_DPI,
                format="png",
                color="#1F2937",
            )
        return path

    def inline(self, expression: str, font_size: float) -> str:
        path = self._render(expression, font_size)
        with PillowImage.open(path) as image:
            width_px, height_px = image.size
        natural_height = height_px * 72 / MATH_DPI
        height = min(max(natural_height, font_size * 0.88), font_size * 1.45)
        width = width_px / height_px * height
        return (
            f'<img src="{html.escape(path.as_posix(), quote=True)}" '
            f'width="{width:.2f}" height="{height:.2f}" valign="-2"/>'
        )

    def display(
        self,
        expression: str,
        available_width: float,
        font_size: float = 11.2,
    ) -> KeepTogether:
        path = self._render(expression, font_size)
        with PillowImage.open(path) as image:
            width_px, height_px = image.size
        width = width_px * 72 / MATH_DPI
        height = height_px * 72 / MATH_DPI
        max_width = available_width * 0.92
        if width > max_width:
            scale = max_width / width
            width *= scale
            height *= scale
        rendered = Image(str(path), width=width, height=height)
        rendered.hAlign = "CENTER"
        return KeepTogether(
            [
                Spacer(1, 2),
                rendered,
                Spacer(1, 7),
            ]
        )


def _font_candidates() -> tuple[tuple[Path, ...], ...]:
    return (
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ),
    )


def _register_fonts() -> tuple[str, str, str, str]:
    for regular, bold, italic, mono in _font_candidates():
        if all(path.exists() for path in (regular, bold, italic, mono)):
            pdfmetrics.registerFont(TTFont("WriteupSans", regular))
            pdfmetrics.registerFont(TTFont("WriteupSans-Bold", bold))
            pdfmetrics.registerFont(TTFont("WriteupSans-Italic", italic))
            pdfmetrics.registerFont(TTFont("WriteupMono", mono))
            pdfmetrics.registerFontFamily(
                "WriteupSans",
                normal="WriteupSans",
                bold="WriteupSans-Bold",
                italic="WriteupSans-Italic",
                boldItalic="WriteupSans-Bold",
            )
            return (
                "WriteupSans",
                "WriteupSans-Bold",
                "WriteupSans-Italic",
                "WriteupMono",
            )
    return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier")


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold, italic, mono = _register_fonts()
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "WriteupTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=20,
            leading=24,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "WriteupH1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=14,
            leading=17,
            textColor=DARK,
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "WriteupH2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=14,
            textColor=BLUE,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "WriteupBody",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.1,
            leading=12.6,
            textColor=DARK,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "source": ParagraphStyle(
            "WriteupSource",
            parent=base["BodyText"],
            fontName=italic,
            fontSize=7.5,
            leading=10,
            textColor=MID,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=1,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "WriteupCaption",
            parent=base["BodyText"],
            fontName=italic,
            fontSize=7.6,
            leading=9.5,
            textColor=MID,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "WriteupTable",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=6.8,
            leading=8.4,
            textColor=DARK,
        ),
        "table_header": ParagraphStyle(
            "WriteupTableHeader",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=6.8,
            leading=8.4,
            textColor=colors.white,
        ),
        "list": ParagraphStyle(
            "WriteupList",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9,
            leading=12,
            textColor=DARK,
            spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "WriteupCode",
            parent=base["Code"],
            fontName=mono,
            fontSize=7.8,
            leading=10,
            textColor=DARK,
            backColor=VERY_LIGHT,
            borderColor=LIGHT,
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=6,
        ),
    }
    return styles


def _inline_markup(
    text: str,
    math_renderer: MathRenderer,
    font_size: float,
) -> str:
    placeholders: dict[str, str] = {}

    def protect(value: str) -> str:
        key = f"@@TOKEN{len(placeholders)}@@"
        placeholders[key] = value
        return key

    text = re.sub(
        r"`([^`]+)`",
        lambda match: protect(
            "<font name=\"WriteupMono\">"
            + html.escape(match.group(1))
            + "</font>"
        ),
        text,
    )
    text = re.sub(
        r"(?<!\\)\$([^$\n]+?)(?<!\\)\$",
        lambda match: protect(
            math_renderer.inline(match.group(1), font_size)
        ),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: protect(
            "<a href=\""
            + html.escape(match.group(2), quote=True)
            + "\" color=\"#0072B2\">"
            + html.escape(match.group(1))
            + "</a>"
        ),
        text,
    )
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _paragraph(
    text: str,
    style: ParagraphStyle,
    math_renderer: MathRenderer,
) -> Paragraph:
    return Paragraph(
        _inline_markup(text.strip(), math_renderer, style.fontSize),
        style,
    )


def _table_widths(column_count: int, available: float) -> list[float]:
    if column_count == 3:
        fractions = (0.16, 0.49, 0.35)
    elif column_count == 5:
        fractions = (0.18, 0.22, 0.23, 0.17, 0.20)
    elif column_count == 4:
        fractions = (0.28, 0.24, 0.24, 0.24)
    else:
        fractions = tuple(1 / column_count for _ in range(column_count))
    return [available * fraction for fraction in fractions]


def _make_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available_width: float,
    math_renderer: MathRenderer,
) -> LongTable:
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = (
            styles["table_header"] if row_index == 0 else styles["table"]
        )
        data.append(
            [_paragraph(cell, style, math_renderer) for cell in row]
        )
    table = LongTable(
        data,
        colWidths=_table_widths(len(rows[0]), available_width),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("GRID", (0, 0), (-1, 0), 0.35, BLUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, LIGHT),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    (colors.white, VERY_LIGHT),
                ),
            ]
        )
    )
    return table


def _make_figure(
    path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    available_width: float,
    math_renderer: MathRenderer,
) -> KeepTogether:
    with PillowImage.open(path) as source:
        width_px, height_px = source.size
    max_width = available_width
    max_height = 4.65 * inch
    scale = min(max_width / width_px, max_height / height_px)
    image = Image(
        str(path),
        width=width_px * scale,
        height=height_px * scale,
    )
    image.hAlign = "CENTER"
    return KeepTogether(
        [
            Spacer(1, 4),
            image,
            Paragraph(
                _inline_markup(
                    caption,
                    math_renderer,
                    styles["caption"].fontSize,
                ),
                styles["caption"],
            ),
        ]
    )


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
    )


def _parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        if not _is_separator(lines[index]):
            rows.append(
                [
                    cell.strip()
                    for cell in lines[index].strip().strip("|").split("|")
                ]
            )
        index += 1
    return rows, index


def _parse_list(
    lines: list[str],
    start: int,
    ordered: bool,
) -> tuple[list[str], int]:
    marker = re.compile(r"^\s*\d+\.\s+" if ordered else r"^\s*-\s+")
    items: list[str] = []
    current = ""
    index = start
    while index < len(lines):
        line = lines[index]
        if marker.match(line):
            if current:
                items.append(current.strip())
            current = marker.sub("", line).strip()
        elif line.strip() and current and (
            line.startswith("  ") or not re.match(r"^#{1,3}\s+", line)
        ):
            current += " " + line.strip()
        else:
            break
        index += 1
    if current:
        items.append(current.strip())
    return items, index


def _parse_markdown(
    source: Path,
    styles: dict[str, ParagraphStyle],
    available_width: float,
    math_renderer: MathRenderer,
) -> list[object]:
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines)
        paragraph_lines.clear()
        if text.startswith("*Source:") and text.endswith("*"):
            story.append(
                KeepTogether(
                    [
                        _paragraph(
                            text[1:-1],
                            styles["source"],
                            math_renderer,
                        )
                    ]
                )
            )
        else:
            story.append(
                KeepTogether(
                    [
                        _paragraph(
                            text,
                            styles["body"],
                            math_renderer,
                        )
                    ]
                )
            )

    index = 0
    first_heading = True
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped == "$$":
            flush_paragraph()
            equation_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                equation_lines.append(lines[index].strip())
                index += 1
            if index >= len(lines):
                raise ValueError("Unclosed display-math block in Markdown.")
            expression = " ".join(equation_lines)
            story.append(
                math_renderer.display(expression, available_width)
            )
            index += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            image_path = (source.parent / image_match.group(2)).resolve()
            story.append(
                _make_figure(
                    image_path,
                    image_match.group(1),
                    styles,
                    available_width,
                    math_renderer,
                )
            )
            index += 1
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading = heading_match.group(2)
            if level == 1 and first_heading:
                story.append(
                    _paragraph(heading, styles["title"], math_renderer)
                )
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=1.2,
                        color=ORANGE,
                        spaceAfter=10,
                    )
                )
                first_heading = False
            elif level == 2:
                if heading == "4. Results":
                    story.append(PageBreak())
                elif heading == "References":
                    story.append(PageBreak())
                story.append(
                    _paragraph(heading, styles["h1"], math_renderer)
                )
            else:
                if re.match(r"4\.[2-6]\s", heading):
                    story.append(PageBreak())
                story.append(
                    _paragraph(heading, styles["h2"], math_renderer)
                )
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            rows, index = _parse_table(lines, index)
            story.append(
                _make_table(
                    rows,
                    styles,
                    available_width,
                    math_renderer,
                )
            )
            story.append(Spacer(1, 7))
            continue

        if re.match(r"^\s*-\s+", line):
            flush_paragraph()
            items, index = _parse_list(lines, index, ordered=False)
            story.append(
                ListFlowable(
                    [
                        ListItem(
                            _paragraph(item, styles["list"], math_renderer),
                            leftIndent=12,
                        )
                        for item in items
                    ],
                    bulletType="bullet",
                    start="circle",
                    leftIndent=20,
                    bulletFontName="WriteupSans",
                    bulletFontSize=7,
                    spaceAfter=6,
                )
            )
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            flush_paragraph()
            items, index = _parse_list(lines, index, ordered=True)
            story.append(
                ListFlowable(
                    [
                        ListItem(
                            _paragraph(item, styles["list"], math_renderer),
                            leftIndent=14,
                        )
                        for item in items
                    ],
                    bulletType="1",
                    leftIndent=22,
                    bulletFontName="WriteupSans",
                    bulletFontSize=8,
                    spaceAfter=6,
                )
            )
            continue

        if stripped.startswith("`") and stripped.endswith("`"):
            flush_paragraph()
            story.append(
                Paragraph(
                    html.escape(stripped[1:-1]),
                    styles["code"],
                )
            )
            index += 1
            continue

        paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    return story


def _footer(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFillColor(MID)
    canvas.setFont("WriteupSans", 7.2)
    if document.page > 1:
        canvas.drawString(
            document.leftMargin,
            LETTER[1] - 0.36 * inch,
            "LOO numerical experiments",
        )
    canvas.drawCentredString(
        LETTER[0] / 2,
        0.35 * inch,
        str(document.page),
    )
    canvas.restoreState()


def build_pdf(source: Path, output: Path) -> None:
    styles = _styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / "tmp" / "pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title=(
            "Comparing Schedule-Based Low-Rank Functionals with "
            "KSS, BLM, and Borovickova-Shimer"
        ),
        author="",
        subject="Production Monte Carlo methods, results, and interpretation",
    )
    with tempfile.TemporaryDirectory(
        prefix="writeup-math-",
        dir=temporary_root,
    ) as temporary_directory:
        math_renderer = MathRenderer(Path(temporary_directory))
        story = _parse_markdown(
            source,
            styles,
            LETTER[0] - document.leftMargin - document.rightMargin,
            math_renderer,
        )
        document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pdf(args.input.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}.")


if __name__ == "__main__":
    main()
