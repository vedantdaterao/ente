use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ente_ensu::retrieval::{self, RetrievalDb, RetrievedChunk};

use std::sync::Mutex;
use std::path::PathBuf;

use tauri::{AppHandle, Emitter, State, WebviewWindow};
use tauri::async_runtime;

use crate::commands::common::app_data_dir;
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
    app: AppHandle,
    state: State<'_, RetrievalDownloadState>,
) -> Result<(), ApiError> {
    let destination_path = retrieval_db_path(&app)?
        .to_string_lossy()
        .into_owned();

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

#[tauri::command]
pub async fn retrieval_download_embedding_model(
    window: WebviewWindow,
    app: AppHandle,
) -> Result<(), ApiError> {
    let model_dir = app_data_dir(&app)?;

    async_runtime::spawn_blocking(move || {
        retrieval::ensure_embedding_model_ready(
            &model_dir,
            move |progress| {
                let _ = window.emit("retrieval-download-progress", progress);
            },
            || false,
        )
        .map(|_| ())
        .map_err(|e| ApiError::new("retrieval_download_embedding_model", e))
    })
    .await
    .map_err(|_| fs_thread_error())?
}

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
    app: AppHandle,
    state: State<'_, RetrievalState>,
) -> Result<(), ApiError> {
    let db_path = retrieval_db_path(&app)?;
    let model_path = retrieval_model_path(&app)?;

    crate::logging::log("RAG", format!("db={} exists={}", db_path.display(), db_path.exists()));
    crate::logging::log("RAG", format!("model={} exists={}", model_path.display(), model_path.exists()));

    if !db_path.exists() {
        return Err(ApiError::new("retrieval_open", "DB not downloaded"));
    }
    if !model_path.exists() {
        return Err(ApiError::new("retrieval_open", "Embedding model not downloaded"));
    }

    retrieval::register_vec_extension();

    let rdb = RetrievalDb::open(
        db_path.to_str().unwrap(),
        model_path.to_str().unwrap(),
    )
    .map_err(|e| ApiError::new("retrieval_open", e))?;

    *state.inner.lock().unwrap() = Some(rdb);
    Ok(())
}

const RETRIEVAL_DB_FILE_NAME: &str = "simplewiki_index.db";
const EMBEDDING_MODEL_FILE_NAME: &str = "all-MiniLM-L6-v2.Q8_0.gguf";

fn retrieval_db_path(app: &AppHandle) -> Result<PathBuf, ApiError> {
    Ok(app_data_dir(app)?.join(RETRIEVAL_DB_FILE_NAME))
}

fn retrieval_model_path(app: &AppHandle) -> Result<PathBuf, ApiError> {
    Ok(app_data_dir(app)?.join(EMBEDDING_MODEL_FILE_NAME))
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