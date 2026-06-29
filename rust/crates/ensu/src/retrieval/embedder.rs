use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
};

const EMBEDDING_DIM: usize = 384;
const CONTEXT_SIZE: u32 = 512;

/// all-MiniLM-L6-v2 GGUF model
pub struct EmbeddingModel {
    model: LlamaModel,
    backend: LlamaBackend,
}

impl EmbeddingModel {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let backend = LlamaBackend::init().unwrap_or(LlamaBackend {});
        let model = LlamaModel::load_from_file(
            &backend,
            Path::new(model_path),
            &LlamaModelParams::default(),
        )
        .map_err(|e| format!("Failed to load embedding model: {e}"))?;
        Ok(Self { model, backend })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create embedding context: {e}"))?;

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| format!("Tokenize failed: {e}"))?;

        if tokens.is_empty() {
            return Err("Input produced no tokens".to_string());
        }

        let tokens = &tokens[..tokens.len().min(CONTEXT_SIZE as usize - 1)];

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            // Mark last token to compute logits (required even in embedding mode).
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|e| format!("Batch add failed: {e}"))?;
        }

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("Decode failed: {e}"))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| format!("Failed to get embeddings: {e}"))?;

        if embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "Expected {EMBEDDING_DIM}-dim embedding, got {}",
                embedding.len()
            ));
        }

        Ok(embedding.to_vec())
    }
}
