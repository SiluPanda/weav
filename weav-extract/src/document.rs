//! Document parsing: extract text from various document formats.

use weav_core::error::{WeavError, WeavResult};

use crate::types::{DocumentContent, DocumentFormat, InputDocument};

/// Extract plain text from a document, dispatching on format.
pub fn extract_text(doc: &InputDocument) -> WeavResult<String> {
    match doc.format {
        DocumentFormat::PlainText => extract_plain_text(&doc.content),
        DocumentFormat::Pdf => extract_pdf_text(&doc.content),
        DocumentFormat::Docx => extract_docx_text(&doc.content),
        DocumentFormat::Csv => extract_csv_text(&doc.content),
    }
}

fn extract_plain_text(content: &DocumentContent) -> WeavResult<String> {
    match content {
        DocumentContent::Text(s) => Ok(s.clone()),
        DocumentContent::Binary(bytes) => String::from_utf8(bytes.clone()).map_err(|e| {
            WeavError::DocumentParseError(format!("invalid UTF-8 in plain text: {e}"))
        }),
    }
}

#[cfg(feature = "pdf")]
fn extract_pdf_text(content: &DocumentContent) -> WeavResult<String> {
    let bytes = match content {
        DocumentContent::Binary(b) => b.as_slice(),
        DocumentContent::Text(s) => s.as_bytes(),
    };

    // pdf_oxide requires a file path, so write bytes to a unique temp file.
    // Use pid + nanos + thread-id hash to avoid collisions under concurrent requests.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp_dir = std::env::temp_dir();
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = tmp_dir.join(format!("weav_pdf_{}_{}.pdf", std::process::id(), seq,));
    std::fs::write(&tmp_path, bytes).map_err(|e| {
        WeavError::DocumentParseError(format!("failed to write temp PDF file: {e}"))
    })?;

    let result = (|| -> WeavResult<String> {
        let mut doc = pdf_oxide::PdfDocument::open(&tmp_path)
            .map_err(|e| WeavError::DocumentParseError(format!("failed to parse PDF: {e}")))?;

        let page_count = doc.page_count().map_err(|e| {
            WeavError::DocumentParseError(format!("failed to get PDF page count: {e}"))
        })?;

        let mut text = String::new();
        for page_num in 0..page_count {
            match doc.extract_text(page_num) {
                Ok(page_text) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&page_text);
                }
                Err(_) => continue, // Skip pages that fail to extract
            }
        }

        if text.is_empty() {
            return Err(WeavError::DocumentParseError(
                "PDF contains no extractable text".into(),
            ));
        }

        Ok(text)
    })();

    let _ = std::fs::remove_file(&tmp_path);
    result
}

#[cfg(not(feature = "pdf"))]
fn extract_pdf_text(_content: &DocumentContent) -> WeavResult<String> {
    Err(WeavError::Internal(
        "PDF support not available - enable the 'pdf' feature".into(),
    ))
}

#[cfg(feature = "pdf")]
fn extract_docx_text(content: &DocumentContent) -> WeavResult<String> {
    let bytes = match content {
        DocumentContent::Binary(b) => b.as_slice(),
        DocumentContent::Text(s) => s.as_bytes(),
    };

    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| WeavError::DocumentParseError(format!("failed to open DOCX as ZIP: {e}")))?;

    let mut xml_content = String::new();
    {
        let mut file = archive.by_name("word/document.xml").map_err(|e| {
            WeavError::DocumentParseError(format!("DOCX missing word/document.xml: {e}"))
        })?;
        std::io::Read::read_to_string(&mut file, &mut xml_content)
            .map_err(|e| WeavError::DocumentParseError(format!("failed to read DOCX XML: {e}")))?;
    }

    // Extract text content from XML: collect text between <w:t> tags.
    let text = extract_text_from_docx_xml(&xml_content);

    if text.trim().is_empty() {
        return Err(WeavError::DocumentParseError(
            "DOCX contains no extractable text".into(),
        ));
    }

    Ok(text)
}

#[cfg(not(feature = "pdf"))]
fn extract_docx_text(_content: &DocumentContent) -> WeavResult<String> {
    Err(WeavError::Internal(
        "DOCX support not available - enable the 'pdf' feature".into(),
    ))
}

/// Simple XML text extractor for DOCX word/document.xml.
/// Extracts text from <w:t> elements and adds paragraph breaks.
#[cfg(feature = "pdf")]
fn extract_text_from_docx_xml(xml: &str) -> String {
    let mut result = String::new();
    let mut in_paragraph = false;
    let mut paragraph_text = String::new();

    // Simple state-machine parser for <w:p>, <w:r>, <w:t> elements.
    let mut chars = xml.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '<' {
            let mut tag = String::new();
            for c in chars.by_ref() {
                if c == '>' {
                    break;
                }
                tag.push(c);
            }

            if tag.starts_with("w:p ") || tag == "w:p" {
                in_paragraph = true;
                paragraph_text.clear();
            } else if tag == "/w:p" {
                if in_paragraph && !paragraph_text.trim().is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(paragraph_text.trim());
                }
                in_paragraph = false;
            } else if (tag.starts_with("w:t ") || tag == "w:t") && in_paragraph {
                // Read text content until </w:t>
                let mut text = String::new();
                for c in chars.by_ref() {
                    if c == '<' {
                        // Consume closing tag
                        for c2 in chars.by_ref() {
                            if c2 == '>' {
                                break;
                            }
                        }
                        break;
                    }
                    text.push(c);
                }
                paragraph_text.push_str(&text);
            }
        }
    }

    // Flush any remaining paragraph
    if in_paragraph && !paragraph_text.trim().is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(paragraph_text.trim());
    }

    result
}

fn extract_csv_text(content: &DocumentContent) -> WeavResult<String> {
    let data = match content {
        DocumentContent::Text(s) => s.clone(),
        DocumentContent::Binary(b) => String::from_utf8(b.clone())
            .map_err(|e| WeavError::DocumentParseError(format!("invalid UTF-8 in CSV: {e}")))?,
    };

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(data.as_bytes());

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| WeavError::DocumentParseError(format!("failed to read CSV headers: {e}")))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    if headers.is_empty() {
        return Err(WeavError::DocumentParseError("CSV has no headers".into()));
    }

    let mut sentences = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| {
            WeavError::DocumentParseError(format!("failed to read CSV record: {e}"))
        })?;
        let parts: Vec<String> = headers
            .iter()
            .zip(record.iter())
            .filter(|(_, v)| !v.is_empty())
            .map(|(h, v)| format!("{h}: {v}"))
            .collect();
        if !parts.is_empty() {
            sentences.push(parts.join(", ") + ".");
        }
    }

    if sentences.is_empty() {
        return Err(WeavError::DocumentParseError(
            "CSV contains no data rows".into(),
        ));
    }

    Ok(sentences.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_plain_text_from_text() {
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::PlainText,
            content: DocumentContent::Text("Hello, world!".into()),
        };
        let text = extract_text(&doc).unwrap();
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_extract_plain_text_from_binary() {
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::PlainText,
            content: DocumentContent::Binary(b"Hello bytes".to_vec()),
        };
        let text = extract_text(&doc).unwrap();
        assert_eq!(text, "Hello bytes");
    }

    #[test]
    fn test_extract_plain_text_invalid_utf8() {
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::PlainText,
            content: DocumentContent::Binary(vec![0xFF, 0xFE]),
        };
        let err = extract_text(&doc).unwrap_err();
        assert!(matches!(err, WeavError::DocumentParseError(_)));
    }

    #[test]
    fn test_extract_csv_text() {
        let csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA\n";
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::Csv,
            content: DocumentContent::Text(csv_data.into()),
        };
        let text = extract_text(&doc).unwrap();
        assert!(text.contains("name: Alice"));
        assert!(text.contains("age: 30"));
        assert!(text.contains("city: NYC"));
        assert!(text.contains("name: Bob"));
    }

    #[test]
    fn test_extract_csv_empty_values_skipped() {
        let csv_data = "name,age,city\nAlice,,NYC\n";
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::Csv,
            content: DocumentContent::Text(csv_data.into()),
        };
        let text = extract_text(&doc).unwrap();
        assert!(text.contains("name: Alice"));
        assert!(text.contains("city: NYC"));
        assert!(!text.contains("age:"));
    }

    #[test]
    fn test_extract_csv_no_data() {
        let csv_data = "name,age\n";
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::Csv,
            content: DocumentContent::Text(csv_data.into()),
        };
        let err = extract_text(&doc).unwrap_err();
        assert!(matches!(err, WeavError::DocumentParseError(_)));
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn test_extract_pdf_invalid_bytes() {
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::Pdf,
            content: DocumentContent::Binary(b"not a pdf".to_vec()),
        };
        let err = extract_text(&doc).unwrap_err();
        assert!(matches!(err, WeavError::DocumentParseError(_)));
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn test_extract_docx_invalid_bytes() {
        let doc = InputDocument {
            document_id: "test".into(),
            format: DocumentFormat::Docx,
            content: DocumentContent::Binary(b"not a docx".to_vec()),
        };
        let err = extract_text(&doc).unwrap_err();
        assert!(matches!(err, WeavError::DocumentParseError(_)));
    }
}
