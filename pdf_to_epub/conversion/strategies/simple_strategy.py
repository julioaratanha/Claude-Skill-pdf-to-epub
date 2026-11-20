# ADAPTABLE: Can be extended or used as template for new strategies
# See ~/.claude/skills/pdf-to-epub/reference/code-adaptation.md
"""Simple conversion strategy for single-column fiction books."""

from pathlib import Path
from typing import List, Tuple, Dict
import re
from collections import Counter
import fitz  # PyMuPDF

from .base_strategy import BaseStrategy
from conversion.models import (
    StructuredContent,
    ImageResource,
    BookMetadata,
    Chapter,
    ImageOptimizationConfig,
)
from core.pdf_extractor import PDFExtractor
from conversion.detectors.models import TextBlock
from conversion.detectors.reading_order.y_sorter import YSorter
from conversion.detectors.reading_order.xy_cut_sorter import XYCutSorter
from conversion.detectors.font_analyzer import FontAnalyzer
from conversion.detectors.structure_classifier import StructureClassifier
from conversion.detectors.structure_builder import StructureBuilder
from conversion.detectors.footnote_detector import FootnoteDetector
from conversion.detectors.endnote_formatter import EndnoteFormatter


class SimpleStrategy(BaseStrategy):
    """
    Conversion strategy for simple single-column books (fiction, novels).
    
    Uses:
    - PDFExtractor for text extraction
    - Y-sort for reading order (top-to-bottom)
    - FontAnalyzer + StructureClassifier for chapter detection
    """
    
    def extract(self, pdf_path: Path, config) -> Tuple[List[TextBlock], List[ImageResource], BookMetadata]:
        """
        Extract text blocks, images, and metadata from PDF.
        
        Args:
            pdf_path: Path to the input PDF file
            config: Conversion configuration
            
        Returns:
            Tuple of (text_blocks, images, metadata)
        """
        # Step 1: Extract text blocks using PDFExtractor
        with PDFExtractor(pdf_path) as extractor:
            self._keep_margin_noise_only(extractor)
            blocks = extractor.get_structural_blocks()

        # Step 2: Extract images from PDF
        images = self._extract_images(pdf_path)

        # Step 2b: Apply page_ranges (skip / content) and full-page rasterization
        if self._has_page_selection(config):
            doc = fitz.open(str(pdf_path))
            try:
                page_count = doc.page_count
                kept_pages = self._resolve_kept_pages(config, page_count)
                raster_pages = {
                    self._resolve_page(int(n), page_count) for n in self._rasterize_list(config)
                } & kept_pages
                blocks = [b for b in blocks if b.page in kept_pages]
                images = [img for img in images if img.page_num in kept_pages and img.page_num not in raster_pages]
                images.extend(self._render_pages(doc, sorted(raster_pages)))
                if getattr(config, "rasterize_vector_graphics", False) is True:
                    vector_images = self._render_vector_regions(doc, sorted(kept_pages - raster_pages))
                    blocks = [b for b in blocks if not self._inside_any(b, vector_images)]
                    images.extend(vector_images)
            finally:
                doc.close()

        # Step 3: Optimize images if enabled
        if hasattr(config, 'image_optimization') and config.image_optimization.enabled:
            images = self._optimize_images(images, config.image_optimization)

        # Step 4: Extract metadata from PDF
        metadata = self._extract_metadata(pdf_path, config)

        return blocks, images, metadata

    @staticmethod
    def _rasterize_list(config) -> List[int]:
        pages = getattr(config, "rasterize_pages", None)
        return list(pages) if isinstance(pages, (list, tuple)) else []

    def _has_page_selection(self, config) -> bool:
        """True when config restricts pages or asks for rasterized pages."""
        page_ranges = getattr(config, "page_ranges", None)
        skip = getattr(page_ranges, "skip", None)
        content = getattr(page_ranges, "content", None)
        restricts = (isinstance(skip, (list, tuple)) and len(skip) > 0) or (
            isinstance(content, (list, tuple)) and tuple(content) != (1, -1)
        )
        vectors = getattr(config, "rasterize_vector_graphics", False) is True
        return restricts or vectors or bool(self._rasterize_list(config))

    @staticmethod
    def _keep_margin_noise_only(extractor) -> None:
        """
        Drop repeated-line noise patterns that do not sit in the page margins.

        The extractor treats any line repeated on >20% of pages as noise. That is
        right for running headers/footers, but it also erases recurring section
        labels in the body (e.g. "THE PROBLEM", "OUR PROPOSALS" on every chapter).
        """
        patterns = getattr(extractor, "noise_patterns", None)
        doc = getattr(extractor, "doc", None)
        if not isinstance(patterns, list) or not patterns or doc is None:
            return
        try:
            pages = list(doc)[:100]
        except TypeError:
            return

        total = [0] * len(patterns)
        in_margin = [0] * len(patterns)
        for page in pages:
            height = page.rect.height
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    if not text:
                        continue
                    _, y0, _, y1 = line.get("bbox", (0, 0, 0, 0))
                    for i, pattern in enumerate(patterns):
                        if pattern.search(text):
                            total[i] += 1
                            if y1 <= height * 0.1 or y0 >= height * 0.9:
                                in_margin[i] += 1

        extractor.noise_patterns = [
            p for i, p in enumerate(patterns) if total[i] == 0 or in_margin[i] * 2 >= total[i]
        ]

    def _render_vector_regions(
        self, doc, pages: List[int], min_paths: int = 20, gap: float = 12.0
    ) -> List[ImageResource]:
        """
        Render dense clusters of vector drawings (text converted to outlines,
        charts, infographic boxes) as images, since they carry no extractable text.
        """
        rendered: List[ImageResource] = []
        for page_num in pages:
            page = doc[page_num - 1]
            width, height = page.rect.width, page.rect.height
            rects = sorted(
                (d["rect"] for d in page.get_drawings()
                 if d["rect"].height < height * 0.9 and d["rect"].width < width * 0.95),
                key=lambda r: r.y0,
            )
            clusters: List[list] = []  # [rect, path_count]
            for rect in rects:
                if clusters and rect.y0 <= clusters[-1][0].y1 + gap:
                    clusters[-1][0] |= rect
                    clusters[-1][1] += 1
                else:
                    clusters.append([fitz.Rect(rect), 1])

            for idx, (rect, count) in enumerate(c for c in clusters if c[1] >= min_paths and c[0].height >= 15):
                clip = (rect + (-2, -2, 2, 2)) & page.rect
                pix = page.get_pixmap(dpi=200, clip=clip)
                rendered.append(ImageResource(
                    id=f"img-page-{page_num}-vector-{idx + 1}",
                    filename=f"page{page_num:03d}-vector{idx + 1}.png",
                    data=pix.tobytes("png"),
                    format="png",
                    width=pix.width,
                    height=pix.height,
                    page_num=page_num,
                    bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                ))
        return rendered

    @staticmethod
    def _inside_any(block: TextBlock, images: List[ImageResource]) -> bool:
        """True if the block's centre lies inside a rendered region (avoid duplicates)."""
        cx, cy = (block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2
        for img in images:
            if img.page_num == block.page and img.bbox:
                x0, y0, x1, y1 = img.bbox
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return True
        return False

    @staticmethod
    def _resolve_page(page: int, page_count: int) -> int:
        """Convert a 1-indexed page number (negative = from end) to 1-indexed."""
        return page if page > 0 else page_count + 1 + page

    def _resolve_kept_pages(self, config, page_count: int) -> set[int]:
        """Pages left after applying page_ranges.content and page_ranges.skip."""
        all_pages = set(range(1, page_count + 1))
        page_ranges = getattr(config, "page_ranges", None)
        content = getattr(page_ranges, "content", None) or (1, -1)
        skip_list = getattr(page_ranges, "skip", None) or []
        if not isinstance(content, (tuple, list)) or not isinstance(skip_list, (tuple, list)):
            return all_pages  # loosely-typed config (e.g. test doubles): keep everything
        start = max(1, self._resolve_page(int(content[0]), page_count))
        end = min(page_count, self._resolve_page(int(content[1]), page_count))
        kept = set(range(start, end + 1))
        endnotes = getattr(page_ranges, "endnotes", None)
        if isinstance(endnotes, (tuple, list)) and len(endnotes) == 2:
            # Endnote pages usually sit outside the main content range.
            note_start = max(1, self._resolve_page(int(endnotes[0]), page_count))
            note_end = min(page_count, self._resolve_page(int(endnotes[1]), page_count))
            kept.update(range(note_start, note_end + 1))
        skip = {self._resolve_page(int(p), page_count) for p in skip_list}
        return kept - skip

    def _render_pages(self, doc, pages: List[int]) -> List[ImageResource]:
        """Render whole pages to JPEG (used for covers assembled from many layers)."""
        rendered: List[ImageResource] = []
        for page_num in pages:
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=150)
            rect = page.rect
            rendered.append(ImageResource(
                id=f"img-page-{page_num}-render",
                filename=f"page{page_num:03d}.jpeg",
                data=pix.tobytes("jpeg"),
                format="jpeg",
                width=pix.width,
                height=pix.height,
                page_num=page_num,
                bbox=(0.0, 0.0, rect.width, rect.height),
            ))
        return rendered

    def _extract_images(self, pdf_path: Path) -> List[ImageResource]:
        """
        Extract all images from PDF pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of ImageResource objects
        """
        images: List[ImageResource] = []
        doc = fitz.open(str(pdf_path))

        try:
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict").get("blocks", [])
                image_blocks = [b for b in blocks if b.get("type") == 1]
                seen_xrefs = set()

                for img_index, block in enumerate(image_blocks, start=1):
                    xref = block.get("xref")
                    try:
                        if xref:
                            base_image = doc.extract_image(xref)
                            image_data = base_image["image"]
                            ext = base_image.get("ext", "png")
                            width = base_image["width"]
                            height = base_image["height"]
                        else:
                            image_data = block.get("image")
                            ext = block.get("ext", "png")
                            width = int(block.get("width", 0))
                            height = int(block.get("height", 0))

                        if not image_data:
                            continue

                        if ext not in ["png", "jpeg", "jpg", "gif"]:
                            print(
                                f"Warning: Unsupported image format '{ext}' on page {page_num}; skipping."
                            )
                            continue

                        image_resource = ImageResource(
                            id=f"img-page-{page_num}-{img_index}",
                            filename=f"image{len(images)+1:03d}.{ext}",
                            data=image_data,
                            format=ext,
                            width=width,
                            height=height,
                            page_num=page_num,
                            bbox=tuple(block.get("bbox", (0, 0, 0, 0)))
                        )
                        images.append(image_resource)
                        if xref:
                            seen_xrefs.add(xref)
                    except Exception as e:
                        # Skip images that can't be extracted
                        print(f"Warning: Could not extract image {img_index} from page {page_num}: {e}")
                        continue

                # Fallback: extract images without bbox if dict blocks are missing
                if not image_blocks:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list, start=1):
                        xref = img[0]
                        if xref in seen_xrefs:
                            continue
                        try:
                            base_image = doc.extract_image(xref)
                            ext = base_image.get("ext", "png")
                            if ext not in ["png", "jpeg", "jpg", "gif"]:
                                print(
                                    f"Warning: Unsupported image format '{ext}' on page {page_num}; skipping."
                                )
                                continue

                            image_resource = ImageResource(
                                id=f"img-page-{page_num}-fallback-{img_index}",
                                filename=f"image{len(images)+1:03d}.{ext}",
                                data=base_image["image"],
                                format=ext,
                                width=base_image["width"],
                                height=base_image["height"],
                                page_num=page_num,
                                bbox=None
                            )
                            images.append(image_resource)
                        except Exception as e:
                            print(f"Warning: Could not extract image {img_index} from page {page_num}: {e}")
                            continue
        finally:
            doc.close()

        return images

    def _optimize_images(
        self,
        images: List[ImageResource],
        config: ImageOptimizationConfig
    ) -> List[ImageResource]:
        """
        Optimize extracted images for EPUB.

        Args:
            images: List of ImageResource objects
            config: Image optimization configuration

        Returns:
            List of optimized ImageResource objects
        """
        from core.image_optimizer import (
            ImageOptimizer,
            ImageOptimizationConfig as OptConfig
        )

        opt_config = OptConfig(
            max_width=config.max_width,
            max_height=config.max_height,
            jpeg_quality=config.jpeg_quality,
            convert_png_to_jpeg=config.convert_png_to_jpeg
        )

        optimizer = ImageOptimizer(opt_config)
        return optimizer.optimize_batch(images)

    def _extract_metadata(self, pdf_path: Path, config) -> BookMetadata:
        """
        Extract metadata from PDF info dict.
        
        Falls back to config values or "Unknown" if metadata is not available.
        
        Args:
            pdf_path: Path to the PDF file
            config: Conversion configuration
            
        Returns:
            BookMetadata object
        """
        doc = fitz.open(str(pdf_path))
        
        try:
            pdf_meta = doc.metadata or {}
            
            # Extract metadata with fallback chain: config -> PDF info -> "Unknown"
            title = config.metadata.title if config.metadata.title else (pdf_meta.get('title') or "Unknown")
            author = config.metadata.author if config.metadata.author else (pdf_meta.get('author') or "Unknown")
            language = config.metadata.language or "en"
            publisher = pdf_meta.get('producer')
            description = pdf_meta.get('subject')
            
            metadata = BookMetadata(
                title=title,
                author=author,
                language=language,
                publisher=publisher,
                isbn=None,  # ISBN typically not in PDF metadata
                description=description
            )
        finally:
            doc.close()
        
        return metadata
    
    def order_blocks(self, blocks: List[TextBlock], config) -> Tuple[List[TextBlock], float]:
        """
        Order text blocks using simple Y-sort (top-to-bottom).
        
        Args:
            blocks: List of extracted text blocks
            config: Conversion configuration
            
        Returns:
            Tuple of (ordered blocks, confidence=1.0)
        """
        strategy = getattr(config, "reading_order_strategy", "y_sort")
        if strategy in ["xy_cut", "column_based"]:
            sorter = XYCutSorter()
        else:
            sorter = YSorter()
        ordered_blocks = sorter.sort_blocks(blocks)
        ordered_blocks = self._reorder_side_blocks(ordered_blocks)
        ordered_blocks = self._clean_text(ordered_blocks)

        # Y-sort is deterministic, so confidence is always 1.0
        confidence = 1.0
        
        return ordered_blocks, confidence

    # Letters incl. accented ones; hyphenated compounds kept as one token.
    _WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
    # Line-break hyphenation left by the extractor: "territóri- os".
    _BROKEN_WORD_RE = re.compile(r"([^\W\d_]+)-\s+([^\W\d_A-Z][^\W\d_]*)")
    # Drop cap glued to the end of the preceding short line: "... Renan Santos O".
    _TRAILING_DROP_CAP_RE = re.compile(r"^(.{1,60}?[^\s.:;!?])\s+([A-ZÀ-Ý])$")
    # Suspended hyphens ("first- and second-person") must stay as they are.
    _SUSPENDED_HYPHEN_WORDS = frozenset({"and", "or", "nor", "e", "ou", "y", "und", "oder", "et"})

    def _clean_text(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Repair typographic artifacts using the document's own vocabulary:
        - drop caps extracted as a separate one-letter block ("E" + "sse capítulo")
        - words hyphenated at line breaks ("territóri- os" -> "territórios")
        """
        if not blocks:
            return blocks

        self._vocab: Counter[str] = Counter()
        for block in blocks:
            for token in self._WORD_RE.findall(block.text):
                self._vocab[token.lower()] += 1
                if "-" in token:
                    self._vocab.update(part.lower() for part in token.split("-"))
        block_texts = Counter(block.text.strip() for block in blocks)

        cleaned: List[TextBlock] = []
        pending_cap: str | None = None
        for i, block in enumerate(blocks):
            text = block.text.strip()
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            next_text = nxt.text.strip() if nxt else ""
            next_starts_word = (
                nxt is not None and bool(next_text) and next_text[0].isalpha() and nxt.page == block.page
            )

            if len(text) == 1 and text.isalpha() and text.isupper() and next_starts_word and len(next_text) > 40:
                pending_cap = text  # standalone drop cap: attach to next block
                continue

            match = self._TRAILING_DROP_CAP_RE.match(text)
            different_style = nxt is not None and (
                nxt.font_name != block.font_name or abs(nxt.font_size - block.font_size) > 0.5
            )
            # Lowercase continuation ("... Renan Santos O" + "futuro é"), or a
            # recurring label that also appears on its own elsewhere
            # ("O PROBLEMA O" + "Sistema Único", while "O PROBLEMA" recurs).
            continues = next_starts_word and (
                next_text[0].islower() or (match is not None and block_texts[match.group(1)] > 0)
            )
            if match and continues and different_style:
                # Short line in another style ending in a lone capital: a drop cap.
                block.text = match.group(1)
                cleaned.append(block)
                pending_cap = match.group(2)
                continue

            if pending_cap:
                block.text = self._attach_drop_cap(pending_cap, text)
                pending_cap = None

            block.text = self.dehyphenate(block.text)
            cleaned.append(block)

        return cleaned

    def _attach_drop_cap(self, cap: str, text: str) -> str:
        first_word = self._WORD_RE.match(text)
        word = first_word.group(0).split("-")[0].lower() if first_word else ""
        vocab = getattr(self, "_vocab", Counter())
        # "E"+"sse" -> "Esse" (joined form is a known word; the fragment only
        # occurs here); "O"+"governo" -> "O governo".
        if word and vocab[(cap + word).lower()] > 0 and vocab[word] <= 1:
            return cap + text
        return f"{cap} {text}"

    def dehyphenate(self, text: str) -> str:
        """Join words split by line-break hyphenation, keeping real compounds."""
        vocab = getattr(self, "_vocab", Counter())

        def join(match):
            left, right = match.group(1), match.group(2)
            if right.lower() in self._SUSPENDED_HYPHEN_WORDS:
                return match.group(0)  # "first- and second-person", "pré- e pós-"
            if vocab[f"{left}-{right}".lower()] > 0:
                return f"{left}-{right}"  # compound seen elsewhere, e.g. "pré-campanha"
            return left + right

        return self._BROKEN_WORD_RE.sub(join, text)

    def _reorder_side_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Move narrow side blocks (e.g., margin notes) after main flow for each page.
        Keeps all content while reducing order disruptions.
        """
        if not blocks:
            return []

        pages = sorted(list(set(b.page for b in blocks)))
        reordered = []

        for page in pages:
            page_blocks = [b for b in blocks if b.page == page]
            if len(page_blocks) < 3:
                reordered.extend(sorted(page_blocks, key=lambda b: (b.y0, b.x0)))
                continue

            min_x = min(b.x0 for b in page_blocks)
            max_x = max(b.x1 for b in page_blocks)
            page_width = max_x - min_x
            if page_width <= 0:
                reordered.extend(sorted(page_blocks, key=lambda b: (b.y0, b.x0)))
                continue

            wide_blocks = [b for b in page_blocks if (b.x1 - b.x0) >= page_width * 0.4]
            reference_blocks = wide_blocks if wide_blocks else page_blocks

            main_left = sorted(b.x0 for b in reference_blocks)[len(reference_blocks) // 2]
            main_right = sorted(b.x1 for b in reference_blocks)[len(reference_blocks) // 2]

            main_blocks = []
            side_blocks = []
            for block in page_blocks:
                width_ratio = (block.x1 - block.x0) / page_width
                is_narrow = width_ratio < 0.5
                is_outside = block.x1 < main_left or block.x0 > main_right

                if is_narrow and is_outside:
                    side_blocks.append(block)
                else:
                    main_blocks.append(block)

            reordered.extend(sorted(main_blocks, key=lambda b: (b.y0, b.x0)))
            reordered.extend(sorted(side_blocks, key=lambda b: (b.y0, b.x0)))

        return reordered
    
    def detect_structure(
        self, 
        blocks: List[TextBlock], 
        config,
        images: List[ImageResource],
        metadata: BookMetadata
    ) -> StructuredContent:
        """
        Detect document structure from ordered blocks.
        
        Uses FontAnalyzer to identify heading fonts, then StructureClassifier
        to classify blocks, and StructureBuilder to group into chapters.
        
        Args:
            blocks: List of ordered text blocks
            config: Conversion configuration
            images: List of extracted images
            metadata: Book metadata
            
        Returns:
            StructuredContent with chapters, metadata, and images
        """
        # Step 1: Analyze fonts to identify heading fonts
        font_analyzer = FontAnalyzer()
        font_analyzer.analyze(blocks)  # Prepares font analysis for classifier
        
        # Step 2: Classify each block as heading or paragraph
        classifier = StructureClassifier()
        classified_blocks = classifier.classify(blocks)
        
        # Step 3: Group into chapters
        builder = StructureBuilder()
        page_ranges = getattr(config, "page_ranges", None)
        detect_endnotes = not (page_ranges is not None and getattr(page_ranges, "endnotes", ()) is None)
        chapters = builder.build_chapters(classified_blocks, detect_endnotes=detect_endnotes)
        
        # Handle case where no chapters were detected
        if not chapters:
            # Create a single chapter with all content
            content_html = "\n".join(f"<p>{block.text}</p>" for block in blocks)
            chapters = [Chapter(
                title="Content",
                level=1,
                content=content_html,
                footnotes=[]
            )]
        
        # Create StructuredContent with extracted metadata and images
        structured = StructuredContent(
            chapters=chapters,
            metadata=metadata,
            reading_order_confidence=0.0,  # Will be set by BaseStrategy
            images=images
        )

        # Build rendered chapters with images interleaved for EPUB output
        structured.rendered_chapters = self._render_chapters_with_images(chapters, images, config)
        
        return structured

    def _render_chapters_with_images(
        self,
        chapters,
        images: List[ImageResource],
        config
    ) -> List[Chapter]:
        image_by_index = self._index_images(images)
        chapter_index_map: Dict[int, int] = {}
        for idx, chapter in enumerate(chapters, start=1):
            self._map_chapter_indices(chapter, idx, chapter_index_map)

        images_by_chapter = self._assign_images_to_chapters(chapters, images, chapter_index_map)
        endnotes_filename = self._get_endnotes_filename(chapters)
        footnote_patterns = getattr(getattr(config, "footnote_processing", None), "patterns", None)

        rendered = []
        for idx, chapter in enumerate(chapters, start=1):
            if getattr(chapter, "is_endnotes", False):
                formatter = EndnoteFormatter()
                html = formatter.format_endnotes(chapter.content_blocks)
                rendered.append(Chapter(title=chapter.title, level=1, content=html, footnotes=[]))
                continue

            chapter_html = self._render_chapter_section(
                chapter=chapter,
                images_by_chapter=images_by_chapter,
                endnotes_filename=endnotes_filename,
                image_by_index=image_by_index,
                chapter_file_index=idx,
                footnote_patterns=footnote_patterns
            )
            rendered.append(
                Chapter(
                    title=chapter.title,
                    level=self._normalize_level(chapter.level),
                    content=chapter_html,
                    footnotes=[]
                )
            )

        return rendered

    def _index_images(self, images: List[ImageResource]) -> Dict[int, ImageResource]:
        def sort_key(img: ImageResource):
            y0 = img.bbox[1] if img.bbox else 0.0
            x0 = img.bbox[0] if img.bbox else 0.0
            return (img.page_num, y0, x0)

        image_by_index = {}
        for idx, img in enumerate(sorted(images, key=sort_key), start=1):
            img.anchor_id = f"img-{idx}"
            img.sequence_num = idx
            image_by_index[idx] = img
        return image_by_index

    def _map_chapter_indices(self, chapter, top_index: int, mapping: Dict[int, int]) -> None:
        mapping[id(chapter)] = top_index
        for sub in getattr(chapter, "subchapters", []):
            self._map_chapter_indices(sub, top_index, mapping)

    def _assign_images_to_chapters(self, chapters, images, chapter_index_map: Dict[int, int]) -> Dict[int, List[ImageResource]]:
        chapter_blocks: List[tuple] = []
        self._collect_chapter_blocks(chapters, chapter_blocks)

        images_by_chapter = {}
        if not chapter_blocks:
            if chapters:
                for img in images:
                    img.chapter_index = 1
                images_by_chapter[id(chapters[0])] = list(images)
            return images_by_chapter

        for img in images:
            best = None
            img_y = img.bbox[1] if img.bbox else 0.0

            for chapter, block in chapter_blocks:
                page_diff = abs(img.page_num - block.original_block.page)
                y_diff = abs(img_y - block.original_block.y0)
                score = page_diff * 10000 + y_diff
                if best is None or score < best[0]:
                    best = (score, chapter)

            if best:
                images_by_chapter.setdefault(id(best[1]), []).append(img)
                img.chapter_index = chapter_index_map.get(id(best[1]), 1)

        return images_by_chapter

    def _collect_chapter_blocks(self, chapters, bucket: list) -> None:
        for chapter in chapters:
            for block in getattr(chapter, "content_blocks", []):
                bucket.append((chapter, block))
            for sub in getattr(chapter, "subchapters", []):
                self._collect_chapter_blocks([sub], bucket)

    def _get_endnotes_filename(self, chapters) -> str | None:
        for idx, chapter in enumerate(chapters, start=1):
            if getattr(chapter, "is_endnotes", False):
                return f"chapter{idx}.xhtml"
        return None

    def _render_chapter_section(
        self,
        chapter,
        images_by_chapter: Dict[int, List[ImageResource]],
        endnotes_filename: str | None,
        image_by_index: Dict[int, ImageResource],
        chapter_file_index: int,
        include_heading: bool = False,
        footnote_patterns: List[str] | None = None
    ) -> str:
        html_parts = []
        if include_heading:
            level = max(2, min(6, chapter.level))
            html_parts.append(f"<h{level}>{self._escape_html(chapter.title)}</h{level}>")

        blocks = getattr(chapter, "content_blocks", [])
        images = images_by_chapter.get(id(chapter), [])
        html_parts.append(
            self._render_blocks_with_images(
                blocks=blocks,
                images=images,
                endnotes_filename=endnotes_filename,
                image_by_index=image_by_index,
                chapter_file_index=chapter_file_index,
                skip_heading_title=chapter.title,
                footnote_patterns=footnote_patterns
            )
        )

        for sub in getattr(chapter, "subchapters", []):
            html_parts.append(
                self._render_chapter_section(
                    chapter=sub,
                images_by_chapter=images_by_chapter,
                endnotes_filename=endnotes_filename,
                image_by_index=image_by_index,
                chapter_file_index=chapter_file_index,
                include_heading=True,
                footnote_patterns=footnote_patterns
            )
            )

        return "\n".join(part for part in html_parts if part)

    def _render_blocks_with_images(
        self,
        blocks,
        images: List[ImageResource],
        endnotes_filename: str | None,
        image_by_index: Dict[int, ImageResource],
        chapter_file_index: int,
        skip_heading_title: str | None = None,
        footnote_patterns: List[str] | None = None
    ) -> str:
        items = []
        for block in blocks:
            items.append(("text", block, block.original_block.page, block.original_block.y0, block.original_block.x0))
        for img in images:
            y0 = img.bbox[1] if img.bbox else 0.0
            x0 = img.bbox[0] if img.bbox else 0.0
            items.append(("image", img, img.page_num, y0, x0))

        items.sort(key=lambda item: (item[2], item[3], item[4], 0 if item[0] == "text" else 1))

        detector = FootnoteDetector(patterns=footnote_patterns) if endnotes_filename else None
        html_parts = []
        current_paragraph = ""
        prev_text_block = None

        def flush_paragraph():
            nonlocal current_paragraph
            if not current_paragraph:
                return
            # Paragraphs joined across blocks can carry "word- next" splits too.
            escaped = self._escape_html(self.dehyphenate(current_paragraph.strip()))
            escaped = self._link_figure_refs(escaped, image_by_index, chapter_file_index)
            if detector:
                escaped = detector.convert_to_hyperlinks(escaped, endnotes_filename)
            html_parts.append(f"<p>{escaped}</p>")
            current_paragraph = ""

        for kind, payload, _, _, _ in items:
            if kind == "image":
                flush_paragraph()
                html_parts.append(self._render_image_html(payload))
                continue

            text = payload.original_block.text.strip()
            if not text:
                continue

            if payload.role.startswith("h") and skip_heading_title and text == skip_heading_title:
                continue

            if payload.role.startswith("h") and payload.role[1:].isdigit():
                flush_paragraph()
                level = max(2, min(6, int(payload.role[1:])))
                html_parts.append(f"<h{level}>{self._escape_html(text)}</h{level}>")
                prev_text_block = payload.original_block
                continue

            if not current_paragraph:
                current_paragraph = text
            elif self._is_continuation(prev_text_block, payload.original_block, current_paragraph, text):
                current_paragraph += " " + text
            else:
                flush_paragraph()
                current_paragraph = text
            prev_text_block = payload.original_block

        flush_paragraph()
        return "\n".join(html_parts)

    def _render_image_html(self, image: ImageResource) -> str:
        anchor_id = getattr(image, "anchor_id", image.id)
        seq = getattr(image, "sequence_num", None)
        alt_text = f"Figure {seq}" if seq else f"Image from page {image.page_num}"
        return (
            f'<figure class="chapter-image" id="{anchor_id}">\n'
            f'  <img src="images/{image.filename}" alt="{alt_text}"/>\n'
            f'</figure>'
        )

    def _link_figure_refs(
        self,
        text: str,
        image_by_index: Dict[int, ImageResource],
        chapter_file_index: int
    ) -> str:
        patterns = [
            re.compile(r'\b(Fig(?:ure)?\.?)\s*(\d{1,3})\b', re.IGNORECASE),
            re.compile(r'\b(Рис(?:\.|унок)?\.?)\s*(\d{1,3})\b', re.IGNORECASE),
        ]

        def replacer(match):
            label = match.group(1)
            num = int(match.group(2))
            image = image_by_index.get(num)
            if not image:
                return match.group(0)
            anchor_id = getattr(image, "anchor_id", image.id)
            target_chapter = getattr(image, "chapter_index", chapter_file_index)
            if target_chapter == chapter_file_index:
                href = f"#{anchor_id}"
            else:
                href = f"chapter{target_chapter}.xhtml#{anchor_id}"
            return f'<a href="{href}" class="figure-ref">{label} {num}</a>'

        linked = text
        for pattern in patterns:
            linked = pattern.sub(replacer, linked)
        return linked

    def _escape_html(self, text: str) -> str:
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))

    def _is_continuation(
        self,
        prev_block: TextBlock | None,
        curr_block: TextBlock | None,
        prev_text: str,
        curr_text: str
    ) -> bool:
        if not prev_text or not curr_text:
            return False

        prev_ends_sentence = prev_text.rstrip()[-1] in ".!?:;"
        curr_starts_lower = curr_text[0].islower()
        prev_ends_hyphen = prev_text.rstrip().endswith("-")

        if prev_ends_hyphen:
            return True
        if not prev_ends_sentence and curr_starts_lower:
            return True

        if not prev_block or not curr_block:
            return False

        gap = max(0.0, curr_block.y0 - prev_block.y1)
        font_size = max(prev_block.font_size, curr_block.font_size, 1.0)
        line_height = font_size * 1.25
        indent_delta = curr_block.x0 - prev_block.x0
        indent_threshold = font_size * 1.2

        if gap > line_height * 1.5:
            return False
        if indent_delta > indent_threshold * 1.5 and gap >= line_height * 0.6:
            return False

        return gap <= line_height * 0.75 and abs(indent_delta) <= indent_threshold

    def _normalize_level(self, level: int) -> int:
        return max(1, min(3, level))
