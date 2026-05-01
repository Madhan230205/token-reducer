use pyo3::prelude::*;

/// Fast token count estimator: max(1, int(word_count * 1.3))
/// Matches Python: max(1, int(len(text.split()) * 1.3))
#[pyfunction]
fn estimate_tokens(text: &str) -> PyResult<usize> {
    let count = text.split_whitespace().count();
    Ok((count * 13 / 10).max(1))
}

/// Tokenize text to lowercase alphanumeric+underscore tokens.
/// Matches Python: re.findall(r"[a-zA-Z0-9_]+", text.lower())
#[pyfunction]
fn tokenize(text: &str) -> PyResult<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            for lc in c.to_lowercase() {
                current.push(lc);
            }
        } else if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}

/// Character n-grams with ^ and $ padding.
/// Matches Python:
///   padded = f"^{token}$"
///   if len(padded) < n: return [padded]
///   return [padded[i:i+n] for i in range(len(padded) - n + 1)]
#[pyfunction]
#[pyo3(signature = (token, n=3))]
fn char_ngrams(token: &str, n: usize) -> PyResult<Vec<String>> {
    // All tokens from `tokenize` are ASCII, so bytes == chars here.
    let padded = format!("^{}$", token);
    let bytes = padded.as_bytes();

    if bytes.len() < n {
        return Ok(vec![padded]);
    }

    let grams: Vec<String> = (0..=(bytes.len() - n))
        .map(|i| {
            // SAFETY: padded contains only ASCII (token is [a-z0-9_], ^ and $ are ASCII)
            std::str::from_utf8(&bytes[i..i + n]).unwrap().to_string()
        })
        .collect();

    Ok(grams)
}

/// Split text into overlapping word-count-bounded chunks.
/// Matches Python implementation in chunker.py exactly.
#[pyfunction]
fn chunk_text(text: &str, chunk_size_words: usize, overlap_words: usize) -> PyResult<Vec<String>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut cursor: usize = 0;

    while cursor < lines.len() {
        let mut current: Vec<&str> = Vec::new();
        let mut words_in_chunk: usize = 0;
        let mut end = cursor;

        while end < lines.len() {
            let line = lines[end];
            let line_words = line.split_whitespace().count().max(1);
            if !current.is_empty() && words_in_chunk + line_words > chunk_size_words {
                break;
            }
            current.push(line);
            words_in_chunk += line_words;
            end += 1;
        }

        let chunk = current.join("\n");
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }

        if end >= lines.len() {
            break;
        }

        // Rewind to find overlap start
        let mut rewind_words: usize = 0;
        let mut rewind_cursor: isize = end as isize - 1;
        while rewind_cursor >= cursor as isize && rewind_words < overlap_words {
            let line = lines[rewind_cursor as usize];
            rewind_words += line.split_whitespace().count().max(1);
            rewind_cursor -= 1;
        }

        cursor = ((rewind_cursor + 1) as usize).max(cursor + 1);
    }

    Ok(chunks)
}

#[pymodule]
fn token_reducer_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(char_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
