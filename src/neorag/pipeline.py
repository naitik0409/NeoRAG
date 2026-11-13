from __future__ import annotations

from dataclasses import dataclass

from neorag.agent.react import RAGAgent
from neorag.agent.tools import default_tools
from neorag.config import NeoRAGConfig
from neorag.embeddings.embedder import Embedder
from neorag.llm.base import build_llm
from neorag.retrieval.reranker import CrossEncoderReranker
from neorag.retrieval.retriever import Retriever
from neorag.store.vector_store import VectorStore


@dataclass
class Components:
    cfg: NeoRAGConfig
    llm: object
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    agent: RAGAgent
    web_searcher: object = None


def build_components(cfg: NeoRAGConfig, load_llm: bool = True) -> Components:
    embedder = Embedder(
        model_id=cfg.embeddings.model_id,
        device=cfg.embeddings.device,
        batch_size=cfg.embeddings.batch_size,
        normalize=cfg.embeddings.normalize,
        max_length=cfg.embeddings.max_length,
    )
    store = VectorStore(cfg.resolve(cfg.store.persist_dir), cfg.store.collection)
    reranker = None
    if cfg.retrieval.reranker == "cross-encoder":
        reranker = CrossEncoderReranker(cfg.retrieval.rerank_model, device=cfg.embeddings.device)
    retriever = Retriever(
        embedder=embedder,
        store=store,
        top_k=cfg.retrieval.top_k,
        final_k=cfg.retrieval.final_k,
        reranker=reranker,
        min_score=cfg.retrieval.min_score,
    )
    web_searcher = None
    if cfg.web.enabled:
        from neorag.web.websearch import build_web_searcher

        web_searcher = build_web_searcher(cfg, embedder=embedder)
    llm = None
    agent = None
    if load_llm:
        llm = build_llm(cfg.llm)
        tools = default_tools(
            retriever,
            web_searcher=web_searcher,
            search_top_k=cfg.retrieval.final_k,
            web_top_k=cfg.web.top_k,
        )
        agent = RAGAgent(llm=llm, tools=tools, max_steps=cfg.agent.max_steps)
    return Components(
        cfg=cfg, llm=llm, embedder=embedder, store=store, retriever=retriever, agent=agent,
        web_searcher=web_searcher,
    )
