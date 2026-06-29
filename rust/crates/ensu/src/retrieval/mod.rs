mod db_download;
mod embedder;
mod search;

use std::path::{Path, PathBuf};

pub use db_download::download_retrieval_db;
pub use embedder::EmbeddingModel;
pub use search::{retrieve, RetrievedChunk};

/* 
    Register the sqlite-vec extension for the current process.
    Must be called once before any `rusqlite::Connection` is opened.
    Safe to call multiple times, SQLite deduplicates auto-extensions.
*/ 
pub fn register_vec_extension() {
    use rusqlite::ffi::sqlite3_auto_extension;
    use sqlite_vec::sqlite3_vec_init;
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }
}

use crate::inference::{download_llm_model_files, LlmModelDownloadTarget};  

const EMBEDDING_MODEL_URL: &str =  
    "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf";  
  
pub fn ensure_embedding_model_ready(  
    model_dir: &Path,  
    on_progress: impl FnMut(crate::inference::LlmModelDownloadProgress),  
    is_cancelled: impl Fn() -> bool,  
) -> Result<PathBuf, String> {  
    let dest = model_dir.join("all-MiniLM-L6-v2.Q8_0.gguf");  
    if !dest.exists() {  
        download_llm_model_files(  
            vec![LlmModelDownloadTarget {  
                label: "Embedding model".to_string(),  
                url: EMBEDDING_MODEL_URL.to_string(),  
                destination_path: dest.to_string_lossy().into_owned(),  
            }],  
            on_progress,
            is_cancelled, 
        )?;  
    }  
    Ok(dest)  
}

pub struct RetrievalDb {
    conn: rusqlite::Connection,
    model: EmbeddingModel,
}

impl RetrievalDb {
    pub fn open(db_path: &str, model_path: &str) -> Result<Self, String> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| format!("Failed to open retrieval DB: {e}"))?;
        let model = EmbeddingModel::load(model_path)?;
        Ok(Self { conn, model })
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedChunk>, String> {
        search::retrieve(&self.conn, &self.model, query, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieve() {
        // Point these at your actual files
        let db_path = "/home/vedant/Programming/ente/simplewiki-rag/build/simplewiki_index.db";
        let model_path = "/tmp/all-MiniLM-L6-v2.Q8_0.gguf";

        register_vec_extension();

        let rdb = RetrievalDb::open(db_path, model_path).expect("failed to open");

        let results = rdb.retrieve("who invented the telephone?", 5)
            .expect("retrieve failed");

        assert!(!results.is_empty(), "expected at least one result");

        for r in &results {
            println!("[{:.4}] {} — {}...", r.distance, r.title, &r.content[..100.min(r.content.len())]);
        }
    }
}