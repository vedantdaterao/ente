use rusqlite::{Connection, Result};
use zerocopy::AsBytes;

use super::EmbeddingModel;


#[derive(Debug, Clone, serde::Serialize)]
pub struct RetrievedChunk {
    pub article_id: i64,
    pub title: String,
    pub content: String,
    pub distance: f32,
}

/// Find the `limit` most relevant articles for `query` using vec0 KNN search.
///
/// `conn` must have been opened after [`super::register_vec_extension`] was called.
pub fn retrieve(
    conn: &Connection,
    model: &EmbeddingModel,
    query: &str,
    limit: usize,
) -> Result<Vec<RetrievedChunk>, String> {
    let embedding = model.embed(query)?;

    /*
        vec0 KNN syntax: WHERE embedding MATCH <blob> ORDER BY distance LIMIT k
        The embedding must be passed as raw f32 bytes (little-endian, packed).
    */
    let sql = "
        SELECT v.article_id, a.title, a.content, v.distance
        FROM article_vec v
        JOIN article a ON a.id = v.article_id
        WHERE v.embedding MATCH ?1
            AND k = ?2
        ORDER BY v.distance
    ";

    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map(
            rusqlite::params![embedding.as_bytes(), limit as i64],
            |row| {
                Ok(RetrievedChunk {
                    article_id: row.get(0)?,
                    title: row.get(1)?,
                    content: row.get(2)?,
                    distance: row.get(3)?,
                })
            },
        )
        .map_err(|e| e.to_string())?;

    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}