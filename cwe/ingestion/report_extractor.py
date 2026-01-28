"""Report and document extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cwe.models.artifact import ReportArtifact, ReportSection


@dataclass
class ReportExtractorConfig:
    """Configuration for report extraction."""
    
    # Text extraction
    extract_tables: bool = True
    extract_images: bool = False
    
    # Section detection
    detect_sections: bool = True
    
    # Limits
    max_pages: int = 100
    max_text_length: int = 500_000


class ReportExtractor:
    """
    Extracts content from documents and reports.
    
    Supports:
    - PDF documents
    - Word documents (.docx)
    - Plain text files
    """
    
    def __init__(self, config: ReportExtractorConfig | None = None):
        self.config = config or ReportExtractorConfig()
    
    def extract(self, doc_path: Path | str) -> ReportArtifact:
        """
        Extract content from a document.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            ReportArtifact with extracted content
        """
        doc_path = Path(doc_path)
        ext = doc_path.suffix.lower()
        
        if ext == ".pdf":
            return self._extract_pdf(doc_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(doc_path)
        else:
            return self._extract_text(doc_path)
    
    def _extract_pdf(self, pdf_path: Path) -> ReportArtifact:
        """Extract content from a PDF."""
        from pypdf import PdfReader
        
        reader = PdfReader(str(pdf_path))
        
        artifact = ReportArtifact(
            incident_id=None,
            filename=pdf_path.name,
            file_path=str(pdf_path),
            page_count=len(reader.pages),
            document_type="pdf",
        )
        
        # Extract text from each page
        full_text_parts = []
        for page_num, page in enumerate(reader.pages[:self.config.max_pages]):
            text = page.extract_text()
            if text:
                full_text_parts.append(text)
                
                # Create section for each page
                artifact.sections.append(ReportSection(
                    section_id=f"page_{page_num + 1}",
                    title=f"Page {page_num + 1}",
                    page_numbers=[page_num + 1],
                    content=text,
                ))
        
        artifact.full_text = "\n\n".join(full_text_parts)[:self.config.max_text_length]
        
        return artifact
    
    def _extract_docx(self, docx_path: Path) -> ReportArtifact:
        """Extract content from a Word document."""
        from docx import Document
        
        doc = Document(str(docx_path))
        
        artifact = ReportArtifact(
            incident_id=None,
            filename=docx_path.name,
            file_path=str(docx_path),
            document_type="docx",
        )
        
        # Extract paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        artifact.full_text = "\n\n".join(paragraphs)[:self.config.max_text_length]
        
        # Create single section for now
        artifact.sections.append(ReportSection(
            section_id="main",
            content=artifact.full_text,
        ))
        
        return artifact
    
    def _extract_text(self, text_path: Path) -> ReportArtifact:
        """Extract content from a plain text file."""
        with open(text_path, "r", errors="ignore") as f:
            content = f.read()[:self.config.max_text_length]
        
        artifact = ReportArtifact(
            incident_id=None,
            filename=text_path.name,
            file_path=str(text_path),
            document_type="text",
            full_text=content,
        )
        
        artifact.sections.append(ReportSection(
            section_id="main",
            content=content,
        ))
        
        return artifact
