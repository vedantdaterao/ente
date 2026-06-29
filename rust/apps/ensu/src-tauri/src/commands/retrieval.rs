use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ente_ensu::retrieval::{self, RetrievalDb, RetrievedChunk};
use std::sync::Mutex;
use tauri::{Emitter, State, WebviewWindow};
use tauri::async_runtime;

use crate::commands::common::ApiError;

// download state

pub struct RetrievalDownloadState {
    cancel_requested: Arc<AtomicBool>,
}

impl Default for RetrievalDownloadState {
    fn default() -> Self {
        Self {
            cancel_requested: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[tauri::command]
pub async fn retrieval_download_db(
    window: WebviewWindow,
    state: State<'_, RetrievalDownloadState>,
    destination_path: String,
) -> Result<(), ApiError> {
    let cancel_requested = Arc::clone(&state.cancel_requested);
    cancel_requested.store(false, Ordering::SeqCst);

    async_runtime::spawn_blocking(move || {
        retrieval::download_retrieval_db(
            &destination_path,
            move |progress| {
                let _ = window.emit("retrieval-download-progress", progress);
            },
            move || cancel_requested.load(Ordering::SeqCst),
        )
        .map_err(|e| ApiError::new("retrieval_download", &e))
    })
    .await
    .map_err(|_| fs_thread_error())?
}

#[tauri::command]
pub fn retrieval_cancel_download(state: State<'_, RetrievalDownloadState>) {
    state.cancel_requested.store(true, Ordering::SeqCst);
}

// retrieval state

pub struct RetrievalState {
    inner: Mutex<Option<RetrievalDb>>,
}

impl Default for RetrievalState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }
}

/// Open the SQLite retrieval DB and load the embedding model.
/// Must be called before `retrieval_query`.
/// Safe to call again to swap in a different DB or model path.
#[tauri::command]
pub fn retrieval_open(
    state: State<'_, RetrievalState>,
    db_path: String,
    model_path: String,
) -> Result<(), ApiError> {
    retrieval::register_vec_extension();

    let rdb = RetrievalDb::open(&db_path, &model_path)
        .map_err(|e| ApiError::new("retrieval_open", e))?;

    *state.inner.lock().unwrap() = Some(rdb);

    Ok(())
}

/// Query the retrieval DB for the most relevant chunks for `query`.
/// Returns up to `top_k` results ordered by ascending distance (closest first).
#[tauri::command]
pub fn retrieval_query(
    state: State<'_, RetrievalState>,
    query: String,
    top_k: u32,
) -> Result<Vec<RetrievedChunk>, ApiError> {
    let inner = state.inner.lock().unwrap();

    let rdb = inner
        .as_ref()
        .ok_or_else(|| ApiError::new("retrieval_query", "Not open — call retrieval_open first"))?;

    rdb.retrieve(&query, top_k as usize)
        .map_err(|e| ApiError::new("retrieval_query", e))
}

// helpers
fn fs_thread_error() -> ApiError {
    ApiError::new("io_thread", "FS task failed")
}