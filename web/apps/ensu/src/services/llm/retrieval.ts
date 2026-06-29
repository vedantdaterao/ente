import { invoke } from "@tauri-apps/api/core";

export interface RetrievedChunk {
    article_id: number;
    title: string;
    content: string;
    distance: number;
}

export async function retrievalOpen(): Promise<void> {
    await invoke("retrieval_open");
}

export async function retrievalQuery(
    query: string,
    topK: number,
): Promise<RetrievedChunk[]> {
    return invoke("retrieval_query", { query, topK });
}

export function buildRagContext(chunks: RetrievedChunk[]): string {
    return chunks
        .map((c) => `### ${c.title}\n${c.content}`)
        .join("\n\n");
}