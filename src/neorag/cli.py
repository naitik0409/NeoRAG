from __future__ import annotations

import argparse
import json
import sys

from neorag.config import load_config, PROJECT_ROOT
from neorag.env import load_env_file
from neorag.logging_utils import logger


def _load_corpus_and_chunks(cfg):
    from neorag.data.chunker import chunk_documents
    from neorag.data.loaders import load_corpus

    docs = load_corpus(cfg.resolve(cfg.ingest.source_dir), cfg.ingest.file_types)
    chunks = chunk_documents(docs, chunk_size=cfg.ingest.chunk_size, chunk_overlap=cfg.ingest.chunk_overlap)
    return docs, chunks


def _ingest_chunks(comps, chunks, clear: bool = False) -> int:
    if not chunks:
        return 0
    if clear or comps.store.count() == 0:
        comps.store.clear()
    logger.info(f"Embedding {len(chunks)} chunks with {comps.cfg.embeddings.model_id}...")
    vectors = comps.embedder.embed([c.text for c in chunks])
    n = comps.store.add_chunks(chunks, vectors)
    logger.info(f"Ingested {n} chunks. Store total: {comps.store.count()}")
    return n


def cmd_ingest(args) -> int:
    from neorag.pipeline import build_components

    cfg = load_config(args.config)
    comps = build_components(cfg, load_llm=False)
    _, chunks = _load_corpus_and_chunks(cfg)
    if not chunks:
        logger.warning("No chunks produced; put files in the corpus directory first.")
        return 1
    _ingest_chunks(comps, chunks, clear=args.clear)
    return 0


def cmd_research(args) -> int:
    from urllib.parse import urlparse

    from neorag.pipeline import build_components
    from neorag.web.scraper import slugify
    from neorag.web.websearch import build_web_searcher

    cfg = load_config(args.config)
    comps = build_components(cfg, load_llm=False)
    if comps.web_searcher is None:
        comps.web_searcher = build_web_searcher(cfg, embedder=comps.embedder)
    pages = comps.web_searcher.collect_corpus_text(args.query, max_pages=args.pages)
    if not pages:
        logger.warning("No pages could be fetched for this query.")
        return 1
    out_dir = cfg.resolve(cfg.ingest.source_dir) / "web" / slugify(args.query)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pages, start=1):
        domain = slugify(urlparse(page["url"]).netloc)
        path = out_dir / f"{i:03d}_{domain}.md"
        content = (
            f"source: {page['url']}\ntitle: {page['title']}\nquery: {args.query}\n\n{page['text']}\n"
        )
        path.write_text(content, encoding="utf-8")
    logger.info(f"Saved {len(pages)} pages under {out_dir}")
    if not args.no_ingest:
        docs = [type("D", (), {"doc_id": "", "text": p["text"], "metadata": {"source": p["url"], "title": p["title"], "file_type": "md"}})() for p in pages]
        from neorag.data.chunker import chunk_documents

        chunks = chunk_documents(docs, chunk_size=cfg.ingest.chunk_size, chunk_overlap=cfg.ingest.chunk_overlap)
        for c in chunks:
            c.chunk_id = f"web_{c.chunk_id}"
        _ingest_chunks(comps, chunks, clear=False)
    return 0


def _direct_answer(llm, retriever, question: str):
    from neorag.agent.direct import direct_answer

    return direct_answer(llm, retriever, question)


def cmd_chat(args) -> int:
    from neorag.pipeline import build_components
    from neorag.ui import ChatUI

    cfg = load_config(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
    if args.model:
        if cfg.llm.backend == "api":
            cfg.llm.api_model = args.model
        else:
            cfg.llm.model_id = args.model
    if args.api_base:
        cfg.llm.api_base = args.api_base
    comps = build_components(cfg, load_llm=True)
    mode = "agent" if args.agent else "fast"
    ui = ChatUI(comps, mode=mode, web_off=args.direct, show_steps=args.steps)
    ui.run()
    return 0


def cmd_build_dataset(args) -> int:
    from neorag.data.qa_dataset import build_sft_examples, write_jsonl
    from neorag.pipeline import build_components

    cfg = load_config(args.config)
    comps = build_components(cfg, load_llm=True)
    _, chunks = _load_corpus_and_chunks(cfg)
    if not chunks:
        logger.warning("No chunks found; run `neorag ingest` after adding documents.")
        return 1
    examples = build_sft_examples(
        comps.llm,
        chunks,
        examples_per_chunk=args.per_chunk,
        include_trajectories=not args.no_trajectories,
        include_template=args.template_fallback,
        limit=args.limit,
    )
    out_path = cfg.resolve(args.out or cfg.training.dataset_path)
    n = write_jsonl(examples, out_path)
    logger.info(f"Dataset ready at {out_path} ({n} examples)")
    return 0


def cmd_train(args) -> int:
    from neorag.training.sft import train_sft

    cfg = load_config(args.config)
    train_sft(cfg.training)
    return 0


def cmd_merge(args) -> int:
    from neorag.training.sft import merge_adapter

    cfg = load_config(args.config)
    merge_adapter(cfg.training, adapter_dir=args.adapter, merged_dir=args.out)
    return 0


def cmd_evaluate(args) -> int:
    from neorag.eval.metrics import evaluate_agent
    from neorag.pipeline import build_components
    from neorag.training.data_format import load_jsonl

    cfg = load_config(args.config)
    comps = build_components(cfg, load_llm=True)
    records = load_jsonl(cfg.resolve(args.dataset))
    gold = [
        {"question": r["meta"]["question"], "answer": r["meta"]["answer"]}
        for r in records
        if r.get("meta", {}).get("kind") == "direct_qa" and "question" in r.get("meta", {})
    ]
    if not gold:
        logger.warning("No eval records with meta.question/answer found (kind=direct_qa).")
        return 1
    if args.limit:
        gold = gold[: args.limit]
    results = []
    for g in gold:
        if args.direct:
            answer, _ = _direct_answer(comps.llm, comps.retriever, g["question"])
            results.append(
                (g, type("R", (), {"answer": answer, "citations": [1], "passages": {1: None}, "steps_used": 0, "gave_up": False})())
            )
        else:
            results.append((g, comps.agent.run(g["question"])))
    summary = evaluate_agent(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        out_path = cfg.resolve(args.out)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0


def cmd_wipe(args) -> int:
    from neorag.pipeline import build_components

    cfg = load_config(args.config)
    comps = build_components(cfg, load_llm=False)
    comps.store.clear()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neorag", description="NeoRAG - free, local, fine-tunable RAG agent")
    parser.add_argument("--config", default=None, help="Path to neorag.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Load, chunk and embed documents into the vector store")
    p.add_argument("--clear", action="store_true", help="Clear the store before ingesting")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser(
        "research", help="Free web research: search, scrape pages, save to corpus and ingest"
    )
    p.add_argument("query", help="Search query")
    p.add_argument("--pages", type=int, default=None, help="Max pages to scrape")
    p.add_argument("--no-ingest", action="store_true", help="Only save pages, skip ingesting")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("chat", help="Interactive RAG chat (rich terminal UI)")
    p.add_argument("--direct", action="store_true", help="Docs-only fast mode (no web)")
    p.add_argument("--agent", action="store_true", help="Start in agent mode (step-by-step ReAct loop)")
    p.add_argument("--steps", action="store_true", help="Show agent step trace")
    p.add_argument("--backend", choices=["local", "api"], default=None)
    p.add_argument("--model", default=None, help="Override model_id (local) or api_model (api)")
    p.add_argument("--api-base", default=None, help="Override llm.api_base")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("build-dataset", help="Generate SFT fine-tuning data from your corpus")
    p.add_argument("--per-chunk", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-trajectories", action="store_true")
    p.add_argument("--template-fallback", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_build_dataset)

    sub.add_parser("train", help="LoRA fine-tune the base model on the SFT dataset").set_defaults(func=cmd_train)

    p = sub.add_parser("merge", help="Merge the LoRA adapter into a standalone model")
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("evaluate", help="Evaluate the agent on generated QA pairs")
    p.add_argument("--dataset", default="data/finetune/sft.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("wipe", help="Clear the vector store").set_defaults(func=cmd_wipe)
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    load_env_file(PROJECT_ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {exc}")
        if "--debug" in (argv or sys.argv[1:]):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
