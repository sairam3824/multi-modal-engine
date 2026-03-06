from typing import List, Dict, Any

class PDFParser:
    """Parse PDF documents and extract layout elements."""
    
    def parse(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract all elements from PDF with layout analysis."""
        fitz = self._get_fitz_module()
        elements = []
        doc = fitz.open(pdf_path)

        try:
            for page_num, page in enumerate(doc, start=1):
                # Extract text blocks
                blocks = page.get_text("dict")["blocks"]

                for block_idx, block in enumerate(blocks):
                    if block["type"] == 0:  # Text block
                        text = " ".join(
                            [
                                span["text"]
                                for line in block.get("lines", [])
                                for span in line.get("spans", [])
                            ]
                        )
                        if text.strip():
                            elements.append(
                                {
                                    "type": "text",
                                    "content": text.strip(),
                                    "page": page_num,
                                    "bbox": block["bbox"],
                                    "element_id": f"p{page_num}_b{block_idx}",
                                }
                            )

                    elif block["type"] == 1:  # Image block
                        img_data = self._extract_image(page, block)
                        if img_data:
                            elements.append(
                                {
                                    "type": "image",
                                    "content": img_data,
                                    "page": page_num,
                                    "bbox": block["bbox"],
                                    "element_id": f"p{page_num}_img{block_idx}",
                                }
                            )

                # Extract tables when available in current PyMuPDF version
                tables = self._detect_tables(page)
                for table_idx, table in enumerate(tables):
                    elements.append(
                        {
                            "type": "table",
                            "content": table,
                            "page": page_num,
                            "element_id": f"p{page_num}_tbl{table_idx}",
                            "bbox": table.get("bbox"),
                        }
                    )
        finally:
            doc.close()

        return elements

    def _get_fitz_module(self):
        """Import PyMuPDF safely and avoid collisions with unrelated `fitz` package."""
        try:
            import fitz

            return fitz
        except Exception:
            try:
                import pymupdf as fitz

                return fitz
            except Exception as exc:
                raise ImportError(
                    "PyMuPDF is required for PDF parsing. Install `pymupdf` and "
                    "remove conflicting `fitz` package."
                ) from exc
    
    def _extract_image(self, page, block) -> bytes:
        """Extract image bytes from block."""
        try:
            pix = page.get_pixmap(clip=block["bbox"])
            return pix.tobytes("png")
        except:
            return None
    
    def _detect_tables(self, page) -> List[Dict[str, Any]]:
        """Detect and extract tables from page."""
        find_tables = getattr(page, "find_tables", None)
        if not callable(find_tables):
            return []

        try:
            found = find_tables()
            detected = getattr(found, "tables", None)
            if detected is None:
                detected = found or []
        except Exception:
            return []

        tables = []
        for table in detected:
            try:
                raw_rows = table.extract()
            except Exception:
                continue

            if not raw_rows:
                continue

            rows = [
                [cell if cell is not None else "" for cell in row]
                for row in raw_rows
            ]
            if not rows:
                continue

            tables.append({"data": rows, "bbox": getattr(table, "bbox", None)})

        return tables
