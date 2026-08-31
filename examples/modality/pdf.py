"""Pdf: store a PDF, stream a pdfplumber document.

Needs pdfplumber.
"""

from litdata import Pdf, StreamingDataset, optimize

# Minimal one-page PDF (US letter).
_PDF = (
    b"%PDF-1.1\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>endobj\n"
    b"trailer<< /Root 1 0 R >>\n"
    b"%%EOF\n"
)


def make_sample(index: int) -> dict:
    return {
        "index": index,
        # Pdf(path="p.pdf")
        # Pdf(pdf=pdfplumber_doc)
        "doc": Pdf(bytes=_PDF),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(4)),
        output_dir="example_optimize_dataset/pdf",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/pdf")[0]
    pdf = sample["doc"]  # Pdfplumber
    print(len(pdf.pages), pdf.pages[0].width, pdf.pages[0].height)
