"""
Tests for layout-cleanup features of SimpleStrategy / StructureBuilder:
page selection, full-page and vector-graphics rasterization, drop caps,
line-break dehyphenation, endnote opt-out, margin-only noise filtering and
proximity-based heading merging.
"""
import re

import fitz
import pytest

from conversion.strategies.simple_strategy import SimpleStrategy
from conversion.detectors.structure_builder import StructureBuilder
from conversion.detectors.models import TextBlock, SemanticBlock
from conversion.models import (
    ConversionConfig,
    PageRanges,
    ExcludeRegions,
    MultiColumnConfig,
    HeadingConfig,
    FootnoteConfig,
    BookMetadata,
)


def tb(text, *, page=1, y0=0.0, y1=10.0, x0=0.0, x1=400.0, font="Body", size=10.5, flags=0):
    return TextBlock(text=text, page=page, x0=x0, y0=y0, x1=x1, y1=y1,
                     font_name=font, font_size=size, flags=flags)


def make_config(**overrides):
    params = dict(
        page_ranges=PageRanges(),
        exclude_regions=ExcludeRegions(),
        multi_column=MultiColumnConfig(),
        reading_order_strategy="y_sort",
        heading_detection=HeadingConfig(),
        footnote_processing=FootnoteConfig(),
        metadata=BookMetadata(title="T", author="A", language="pt"),
    )
    params.update(overrides)
    return ConversionConfig(**params)


LONG = "x" * 50  # pads blocks past the 40-char drop-cap guard


# ---------------------------------------------------------------- page ranges

class TestPageSelection:
    def test_default_keeps_all_pages(self):
        strategy = SimpleStrategy()
        assert strategy._resolve_kept_pages(make_config(), 5) == {1, 2, 3, 4, 5}
        assert not strategy._has_page_selection(make_config())

    def test_skip_and_negative_content(self):
        config = make_config(page_ranges=PageRanges(skip=[2], content=(1, -2), endnotes=None))
        strategy = SimpleStrategy()
        assert strategy._has_page_selection(config)
        assert strategy._resolve_kept_pages(config, 6) == {1, 3, 4, 5}

    def test_endnote_pages_kept_outside_content(self):
        config = make_config(page_ranges=PageRanges(skip=[1], content=(2, -3), endnotes=(-2, -1)))
        assert SimpleStrategy()._resolve_kept_pages(config, 8) == {2, 3, 4, 5, 6, 7, 8}

    def test_rasterize_pages_alone_triggers_selection(self):
        assert SimpleStrategy()._has_page_selection(make_config(rasterize_pages=[1]))

    def test_from_dict_accepts_new_fields(self):
        config = ConversionConfig.from_dict({
            "page_ranges": {"skip": [2], "content": [1, -1], "endnotes": None},
            "exclude_regions": {},
            "multi_column": {},
            "reading_order_strategy": "y_sort",
            "heading_detection": {},
            "footnote_processing": {},
            "metadata": {"title": "T", "author": "A", "language": "pt"},
            "rasterize_pages": [1, -1],
            "rasterize_vector_graphics": True,
        })
        assert config.page_ranges.endnotes is None
        assert config.rasterize_pages == [1, -1]
        assert config.rasterize_vector_graphics is True


# ------------------------------------------------------------- rasterization

@pytest.fixture
def sample_pdf(tmp_path):
    """3 pages: text, text + dense vector box, text only."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=400, height=600)
        page.insert_text((40, 60), f"Body text on page {i + 1} with enough words.", fontsize=11)
        if i == 1:
            # 30 small filled rectangles in one band: "outlined text" stand-in
            for k in range(30):
                page.draw_rect(fitz.Rect(40 + k * 10, 200, 47 + k * 10, 215), fill=(0, 0, 0))
            page.insert_text((60, 210), "inside box", fontsize=8)
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


class TestRasterization:
    def test_rasterized_page_replaces_its_images(self, sample_pdf):
        config = make_config(rasterize_pages=[-1])
        blocks, images, _ = SimpleStrategy().extract(sample_pdf, config)
        rendered = [img for img in images if img.id.endswith("-render")]
        assert [img.page_num for img in rendered] == [3]
        assert rendered[0].bbox == (0.0, 0.0, 400.0, 600.0)
        assert any(b.page == 3 for b in blocks)  # text kept for search/accessibility

    def test_skipped_pages_drop_blocks(self, sample_pdf):
        config = make_config(page_ranges=PageRanges(skip=[2], content=(1, -1), endnotes=None))
        blocks, _, _ = SimpleStrategy().extract(sample_pdf, config)
        assert {b.page for b in blocks} == {1, 3}

    def test_vector_cluster_rendered_and_inner_text_removed(self, sample_pdf):
        config = make_config(rasterize_vector_graphics=True)
        blocks, images, _ = SimpleStrategy().extract(sample_pdf, config)
        vectors = [img for img in images if "-vector" in img.id]
        assert len(vectors) == 1 and vectors[0].page_num == 2
        assert not any("inside box" in b.text for b in blocks)
        assert any(b.page == 2 and "Body text" in b.text for b in blocks)

    def test_vector_rendering_is_opt_in(self, sample_pdf):
        _, images, _ = SimpleStrategy().extract(sample_pdf, make_config())
        assert not any("-vector" in img.id for img in images)


# ------------------------------------------------- drop caps / hyphenation

class TestTextCleanup:
    def test_drop_cap_joined_without_space_when_fragment_is_not_a_word(self):
        blocks = [
            tb("E", y0=0, y1=40, size=48),
            tb("sse capítulo trata do maior problema " + LONG, y0=0, y1=40),
            tb("Esse é outro parágrafo.", y0=50, y1=60),
        ]
        out = SimpleStrategy()._clean_text(blocks)
        assert len(out) == 2
        assert out[0].text.startswith("Esse capítulo trata")

    def test_drop_cap_joined_with_space_for_article(self):
        blocks = [tb("O", y0=0, y1=40), tb("governo brasileiro hoje não tem dinheiro " + LONG, y0=0, y1=40)]
        out = SimpleStrategy()._clean_text(blocks)
        assert out[0].text.startswith("O governo brasileiro")

    def test_drop_cap_on_other_page_is_left_alone(self):
        blocks = [tb("O", page=1), tb("governo brasileiro " + LONG, page=2)]
        out = SimpleStrategy()._clean_text(blocks)
        assert [b.text for b in out][0] == "O"

    def test_trailing_drop_cap_moved_from_styled_line(self):
        blocks = [
            tb("Uma mensagem de Renan Santos O", font="Italic", size=11.5),
            tb("futuro é glorioso; esse tem sido o mote " + LONG, y0=20, y1=30),
        ]
        out = SimpleStrategy()._clean_text(blocks)
        assert out[0].text == "Uma mensagem de Renan Santos"
        assert out[1].text.startswith("O futuro é glorioso")

    def test_trailing_capital_in_body_text_is_kept(self):
        # English "I" at the end of a body line followed by its continuation.
        blocks = [tb("for example, whenever I"), tb("try to take a more objective look " + LONG, y0=12, y1=22)]
        out = SimpleStrategy()._clean_text(blocks)
        assert out[0].text == "for example, whenever I"

    def test_dehyphenation(self):
        strategy = SimpleStrategy()
        strategy._clean_text([tb("a pré-campanha começou")])  # builds vocabulary
        assert strategy.dehyphenate("territóri- os menores") == "territórios menores"
        assert strategy.dehyphenate("nossa pré- campanha") == "nossa pré-campanha"
        assert strategy.dehyphenate("first- and second-person") == "first- and second-person"
        assert strategy.dehyphenate("pré- e pós-operatório") == "pré- e pós-operatório"
        assert strategy.dehyphenate("Ver- A") == "Ver- A"  # capitalised continuation untouched


# ------------------------------------------------------- noise filtering

class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)


class _FakePage:
    def __init__(self, lines, height=800.0):
        self.rect = fitz.Rect(0, 0, 600, height)
        self._lines = lines

    def get_text(self, kind):
        return {"blocks": [{
            "type": 0,
            "lines": [{"bbox": (0, y, 100, y + 10), "spans": [{"text": t}]} for t, y in self._lines],
        }]}


class _FakeExtractor:
    def __init__(self, patterns, doc):
        self.noise_patterns = patterns
        self.doc = doc


def test_noise_patterns_kept_only_when_in_margins():
    footer = re.compile(r"MY BOOK \d+")
    label = re.compile(r"THE PROBLEM")
    pages = [_FakePage([(f"MY BOOK {i}", 780), ("THE PROBLEM", 300)]) for i in range(5)]
    extractor = _FakeExtractor([footer, label], _FakeDoc(pages))
    SimpleStrategy._keep_margin_noise_only(extractor)
    assert extractor.noise_patterns == [footer]


def test_noise_filter_ignores_mocks():
    extractor = _FakeExtractor(patterns=None, doc=None)
    SimpleStrategy._keep_margin_noise_only(extractor)  # must not raise
    assert extractor.noise_patterns is None


# ---------------------------------------------------- structure builder

def sb(text, role="body", **kw):
    endnote_num = kw.pop("endnote_num", None)
    return SemanticBlock(original_block=tb(text, **kw), role=role, endnote_num=endnote_num)


class TestBuilderOptions:
    def test_endnote_detection_can_be_disabled(self):
        blocks = [sb("Chapter", "h1")] + [
            sb(f"{n}  Proposal number {n}", "endnote", page=n, endnote_num=n) for n in range(1, 8)
        ]
        chapters = StructureBuilder().build_chapters(blocks, detect_endnotes=False)
        assert not any(getattr(c, "is_endnotes", False) for c in chapters)
        assert all(b.role == "body" and b.endnote_num is None for b in chapters[0].content_blocks[1:])

    def test_endnote_detection_on_by_default(self):
        blocks = [sb("Chapter", "h1")] + [
            sb(f"{n}  Note {n}", "endnote", page=n, endnote_num=n) for n in range(1, 8)
        ]
        chapters = StructureBuilder().build_chapters(blocks)
        assert any(getattr(c, "is_endnotes", False) for c in chapters)

    def test_distant_same_level_headings_not_merged(self):
        blocks = [
            sb("O PROBLEMA EM NÚMEROS", "h3", y0=170, y1=180),
            sb("A RESPOSTA DA MISSÃO", "h3", y0=260, y1=270),
        ]
        merged = StructureBuilder()._merge_headers(blocks)
        assert [b.original_block.text for b in merged] == ["O PROBLEMA EM NÚMEROS", "A RESPOSTA DA MISSÃO"]

    def test_adjacent_heading_lines_merged(self):
        blocks = [
            sb("CAPÍTULO I", "h1", y0=58, y1=81, size=23),
            sb("UM REMÉDIO AMARGO", "h1", y0=96, y1=129, size=23),
        ]
        merged = StructureBuilder()._merge_headers(blocks)
        assert [b.original_block.text for b in merged] == ["CAPÍTULO I UM REMÉDIO AMARGO"]


def test_trailing_drop_cap_after_recurring_label_with_capitalised_word():
    blocks = [
        tb("O PROBLEMA", page=1, font="Label", size=9),
        tb("Texto do primeiro capítulo " + LONG, page=1, y0=20, y1=30),
        tb("O PROBLEMA O", page=2, font="Label", size=9),
        tb("Sistema Único de Saúde (SUS) é um sistema " + LONG, page=2, y0=20, y1=30),
    ]
    out = SimpleStrategy()._clean_text(blocks)
    assert out[2].text == "O PROBLEMA"
    assert out[3].text.startswith("O Sistema Único")


def test_heading_ending_in_letter_keeps_it_when_not_recurring():
    blocks = [tb("Apêndice A", font="Label", size=14), tb("Texto do apêndice começa aqui " + LONG, y0=20, y1=30)]
    out = SimpleStrategy()._clean_text(blocks)
    assert out[0].text == "Apêndice A"
