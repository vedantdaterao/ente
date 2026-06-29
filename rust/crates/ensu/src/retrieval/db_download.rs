use std::path::Path;

use reqwest::Client;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Builder;

use crate::inference::{LlmModelDownloadProgress};

const SQLITE_MAGIC: &[u8] = b"SQLite format 3\0";

const SIMPLEWIKI_DB_URL: &str =
    "https://github.com/vedantdaterao/ente/releases/download/retrieval/simplewiki_index.db";

pub fn download_retrieval_db(
    destination_path: &str,
    mut on_progress: impl FnMut(LlmModelDownloadProgress),
    is_cancelled: impl Fn() -> bool,
) -> Result<(), String> {
    let runtime = Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(download_retrieval_db_async(
        destination_path,
        &mut on_progress,
        &is_cancelled,
    ))
}

async fn download_retrieval_db_async(
    destination_path: &str,
    on_progress: &mut impl FnMut(LlmModelDownloadProgress),
    is_cancelled: &impl Fn() -> bool,
) -> Result<(), String> {
    let dest = Path::new(destination_path);
    let tmp = dest.with_extension("db.tmp");

    if dest.exists() {
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }

    let client = Client::new();

    let resume_from = if tmp.exists() {
        tokio::fs::metadata(&tmp)
            .await
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    let head = client
        .head(SIMPLEWIKI_DB_URL)
        .send()
        .await
        .map_err(|e| format!("HEAD failed: {e}"))?;

    let total_bytes: Option<u64> = head
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());

    let accepts_ranges = head
        .headers()
        .get(reqwest::header::ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .map(|v| v != "none")
        .unwrap_or(false);

    let resume_from = if accepts_ranges { resume_from } else { 0 };
    let mut validated_header = resume_from > 0;

    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(resume_from > 0)
        .truncate(resume_from == 0)
        .write(true)
        .open(&tmp)
        .await
        .map_err(|e| format!("Failed to open tmp file: {e}"))?;
    let mut writer = tokio::io::BufWriter::new(file);

    let mut req = client.get(SIMPLEWIKI_DB_URL);
    if resume_from > 0 && accepts_ranges {
        req = req.header(
            reqwest::header::RANGE,
            format!("bytes={resume_from}-"),
        );
    }

    let mut response = req
        .send()
        .await
        .map_err(|e| format!("GET failed: {e}"))?;

    let status = response.status();
    if !status.is_success() && status.as_u16() != 206 {
        return Err(format!("Unexpected status: {status}"));
    }

    let mut downloaded = resume_from;
    let start = std::time::Instant::now();

    loop {
        if is_cancelled() {
            return Err("Download cancelled".to_string());
        }

        let chunk = response
            .chunk()
            .await
            .map_err(|e| format!("Stream error: {e}"))?;

        let Some(chunk) = chunk else { break };

        if !validated_header {
            if chunk.len() < 16 {
                return Err("Response too short to validate SQLite header".to_string());
            }
            if !is_sqlite_header(&chunk) {
                return Err("Not a SQLite database (bad magic bytes)".to_string());
            }
            validated_header = true;
        }

        writer
            .write_all(&chunk)
            .await
            .map_err(|e| format!("Write failed: {e}"))?;

        downloaded += chunk.len() as u64;

        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        let bps = if elapsed.as_secs_f64() > 0.0 {
            downloaded as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let percentage = total_bytes
            .map(|t| downloaded as f64 / t as f64 * 100.0)
            .unwrap_or(0.0);

        on_progress(LlmModelDownloadProgress {
            label: "simplewiki_index.db".to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
            file_downloaded_bytes: downloaded,
            file_total_bytes: total_bytes,
            percentage,
            elapsed_ms,
            bytes_per_second: bps,
            file_elapsed_ms: elapsed_ms,
            file_bytes_per_second: bps,
            retry_count: 0,
            file_retry_count: 0,
            file_complete: false,
            complete: false,
        });
    }

    writer.flush().await.map_err(|e| e.to_string())?;

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis() as u64;
    let bps = if elapsed.as_secs_f64() > 0.0 {
        downloaded as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    on_progress(LlmModelDownloadProgress {
        label: "simplewiki_index.db".to_string(),
        downloaded_bytes: downloaded,
        total_bytes,
        file_downloaded_bytes: downloaded,
        file_total_bytes: total_bytes,
        percentage: 100.0,
        elapsed_ms,
        bytes_per_second: bps,
        file_elapsed_ms: elapsed_ms,
        file_bytes_per_second: bps,
        retry_count: 0,
        file_retry_count: 0,
        file_complete: true,
        complete: true,
    });

    tokio::fs::rename(&tmp, dest)
        .await
        .map_err(|e| format!("Failed to rename tmp to destination: {e}"))?;

    Ok(())
}

fn is_sqlite_header(bytes: &[u8]) -> bool {
    bytes.len() >= 16 && bytes[..16] == *SQLITE_MAGIC
}