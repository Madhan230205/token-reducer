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
        if c.is_ascii_alphanumeric() || c == '_' {
            current.push(c.to_ascii_lowercase());
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
    let padded = format!("^{}$", token);
    let chars: Vec<char> = padded.chars().collect();

    if chars.len() < n {
        return Ok(vec![padded]);
    }

    Ok((0..=(chars.len() - n))
        .map(|i| chars[i..i + n].iter().collect())
        .collect())
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

/// Collapse runs of blank lines to a single blank line (context trimming).
#[pyfunction]
fn collapse_blank_lines(text: &str) -> PyResult<String> {
    let mut out: Vec<&str> = Vec::new();
    let mut prev_blank = false;
    for line in text.lines() {
        let blank = line.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        prev_blank = blank;
        out.push(line);
    }
    Ok(out.join("\n"))
}

/// Hard cap word count for cheap trimming paths (matches compressor semantics loosely).
#[pyfunction]
fn trim_to_max_words(text: &str, max_words: usize) -> PyResult<String> {
    if max_words == 0 {
        return Ok(String::new());
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return Ok(text.to_string());
    }
    Ok(words[..max_words].join(" "))
}

#[pymodule]
fn token_reducer_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(char_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(collapse_blank_lines, m)?)?;
    m.add_function(wrap_pyfunction!(trim_to_max_words, m)?)?;
    Ok(())
}
