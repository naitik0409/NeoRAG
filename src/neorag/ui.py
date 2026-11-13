from __future__ import annotations

import logging
import re
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from neorag.agent.direct import direct_answer
from neorag.agent.fast import fast_answer

BANNER = r"""
  _   _                    ____   ___  _____
 | \ | | ___  _ __  _ __  / ___| / _ \|  ___| __ __ _ _   _  ___ _ __
 |  \| |/ _ \| '_ \| '_ \| |    | | | | |_ | '__/ _` | | | |/ _ \ '__|
 | |\  | (_) | | | | | | | |___| |_| |  _|| | | (_| | |_| |  __/ |
 |_| \_|\___/|_| |_|_| |_|\____|\__\_\_|  |_|  \__,_|\__, |\___|_|
                                                      |___/
"""

HELP_TEXT = """[bold]Commands[/]
  /help        show this help
  /mode        toggle fast mode (one call, hybrid) / agent mode (step-by-step)
  /web         toggle the web_search tool on/off
  /steps       toggle agent step trace visibility
  /stats       show corpus + model stats
  /clear       clear the screen
  /quit        exit"""


class ChatUI:
    def __init__(self, comps, mode: str = "fast", web_off: bool = False, show_steps: bool = False):
        self.comps = comps
        self.cfg = comps.cfg
        self.console = Console()
        self.mode = mode
        self.show_steps = show_steps
        self.web_enabled = bool(self.cfg.web.enabled) and not web_off
        self._web_tool = None
        if comps.agent is not None and "web_search" in comps.agent.tools:
            self._web_tool = comps.agent.tools["web_search"]
        if not self.web_enabled and self._web_tool is not None:
            comps.agent.tools.pop("web_search", None)
        logging.getLogger("neorag").setLevel(logging.ERROR)

    def _model_label(self) -> str:
        llm_cfg = self.cfg.llm
        if llm_cfg.backend == "api":
            return f"{llm_cfg.api_model or llm_cfg.model_id} @ {llm_cfg.api_base}"
        return llm_cfg.model_id

    def _mode_label(self) -> str:
        if self.mode == "agent":
            return "agent (step-by-step ReAct)"
        if self.web_enabled and self.comps.web_searcher is not None:
            return "fast (docs + web, single call)"
        return "fast (docs only, single call)"

    def banner(self) -> None:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_row("Model", self._model_label())
        table.add_row("Backend", self.cfg.llm.backend)
        table.add_row("Embeddings", self.cfg.embeddings.model_id)
        table.add_row("Corpus chunks", str(self.comps.store.count()))
        tools = sorted((self.comps.agent.tools if self.comps.agent else {}).keys()) or ["(none)"]
        table.add_row("Tools", ", ".join(tools))
        table.add_row("Mode", self._mode_label())
        self.console.print(Panel(Text(BANNER.rstrip(), style="bold cyan"), border_style="cyan"))
        self.console.print(Panel(table, title="[bold]NeoRAG ready[/]", border_style="blue"))
        if self.comps.store.count() == 0:
            self.console.print("[yellow]Vector store is empty - run `python -m neorag.cli ingest` first.[/]")
        self.console.print("[dim]Type /help for commands, /quit to exit.[/]\n")

    def _warm_embedder(self) -> None:
        def warm():
            try:
                self.comps.embedder.embed_query("warmup")
            except Exception:
                pass

        threading.Thread(target=warm, daemon=True).start()

    def _spinner_generate(self, label: str, worker_fn):
        result: dict = {}
        errors: list = []

        def worker():
            try:
                result["value"] = worker_fn(cb)
            except Exception as exc:
                errors.append(exc)

        step_text = label

        def cb(info) -> None:
            nonlocal step_text
            if isinstance(info, str):
                step_text = f"{label} - {info}"
            else:
                step_text = f"{label} - {info.get('action', '?')} (step {info.get('step', '?')})"

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with Status(f"[cyan]{step_text}[/]", console=self.console, spinner="dots"):
            while thread.is_alive():
                time.sleep(0.05)
        thread.join()
        if errors:
            raise errors[0]
        return result["value"]

    def _render_steps(self, result) -> None:
        if not self.show_steps or not getattr(result, "trace", None):
            return
        lines = []
        for entry in result.trace:
            lines.append(
                f"[dim]step {entry['step']}: {entry['action']}"
                + (f" - {entry['thought'][:100]}" if entry["thought"] else "")
                + "[/]"
            )
        self.console.print(Text.from_markup("\n".join(lines)))

    def _render_sources(self, result_citations: list[int], passages: dict | list) -> None:
        if isinstance(passages, dict):
            items = [(n, passages[n]) for n in result_citations if n in passages]
        else:
            items = list(enumerate(passages, start=1))
            if not items:
                return
        seen = set()
        table = Table(show_header=False, box=None, pad_edge=False)
        for n, p in items:
            src = str(p.metadata.get("source", "?"))
            if src in seen:
                continue
            seen.add(src)
            origin = p.metadata.get("origin", "docs")
            marker = "web" if origin.startswith("web") else "docs"
            table.add_row(f"[{n}]", f"[dim]{marker}[/]", src)
        self.console.print(table)

    def render_agent_result(self, result) -> None:
        self.console.print()
        self.console.print(
            Panel(
                Markdown(result.answer.strip() or "(empty answer)"),
                title="[bold cyan]NeoRAG[/]",
                border_style="cyan",
            )
        )
        self._render_sources(result.citations, result.passages)
        self._render_steps(result)
        self.console.print(f"[dim]steps={result.steps_used} gave_up={result.gave_up}[/]\n")

    def render_fast(self, answer: str, passages) -> None:
        self.console.print()
        cleaned = re.sub(r"【(\d+)†[^】]*】", r"[\1]", answer or "")
        self.console.print(
            Panel(Markdown(cleaned.strip() or "(empty answer)"), title="[bold cyan]NeoRAG[/]", border_style="cyan")
        )
        cited: set[int] = set()
        for a, b in re.findall(r"\[(\d{1,2})\]|【(\d{1,2})†", cleaned):
            for group in (a, b):
                if group:
                    cited.add(int(group))
        cited = sorted(c for c in cited if 1 <= c <= len(passages))
        cited = [c for c in cited if 1 <= c <= len(passages)]
        if not cited and passages:
            cited = list(range(1, min(3, len(passages)) + 1))
        self._render_sources(cited, passages)
        self.console.print()

    def _friendly_error(self, exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        key_env = self.cfg.llm.api_key_env
        if "missing api key" in lowered or "401" in text or "unauthorized" in lowered or "api key" in lowered:
            return (
                f"[bold red]API key missing or rejected ({key_env}).[/]\n"
                f"[yellow]Paste your free Groq key into the .env file at the project root:[/]\n"
                f"[yellow]  {key_env}=gsk_...   (get one free at console.groq.com)[/]\n"
                f"[dim]{text[:300]}[/]"
            )
        if "429" in text or "rate limit" in lowered:
            return "[yellow]Rate limited. Wait a bit and ask again.[/]"
        if "getaddrinfo" in lowered or "connection" in lowered or "timed out" in lowered:
            return "[yellow]Network problem reaching the API or the web. Check connection and try again.[/]"
        return f"[bold red]{type(exc).__name__}[/] {text[:500]}"

    def _handle_command(self, cmd: str) -> bool:
        cmd = cmd.strip().lower()
        if cmd in {"/quit", "/exit", "quit", "exit"}:
            return False
        if cmd == "/help":
            self.console.print(HELP_TEXT + "\n")
        elif cmd == "/mode":
            self.mode = "agent" if self.mode == "fast" else "fast"
            self.console.print(f"[green]Mode: {self._mode_label()}[/]\n")
        elif cmd == "/steps":
            self.show_steps = not self.show_steps
            self.console.print(f"[green]Step trace: {self.show_steps}[/]\n")
        elif cmd == "/web":
            if self._web_tool is None:
                self.console.print("[yellow]Web tool unavailable (web.enabled=false in config).[/]\n")
            else:
                if self.web_enabled:
                    self.comps.agent.tools.pop("web_search", None)
                    self.web_enabled = False
                else:
                    self.comps.agent.tools["web_search"] = self._web_tool
                    self.web_enabled = True
                self.console.print(f"[green]web_search tool: {'on' if self.web_enabled else 'off'}[/]\n")
        elif cmd == "/stats":
            table = Table(show_header=False, box=None, pad_edge=False)
            table.add_row("Corpus chunks", str(self.comps.store.count()))
            table.add_row("Model", self._model_label())
            table.add_row("Tools", ", ".join(sorted(self.comps.agent.tools.keys())))
            table.add_row("Mode", self._mode_label())
            self.console.print(Panel(table, title="Stats", border_style="blue") + "\n")
        elif cmd == "/clear":
            self.console.clear()
            self.banner()
        else:
            self.console.print(f"[yellow]Unknown command {cmd}. /help for list.[/]\n")
        return True

    def ask(self, question: str) -> None:
        try:
            if self.mode == "fast":
                web_searcher = self.comps.web_searcher if self.web_enabled else None
                answer, passages = self._spinner_generate(
                    "Working",
                    lambda cb: fast_answer(
                        self.comps.llm,
                        self.comps.retriever,
                        web_searcher,
                        question,
                        max_context_chars=self.cfg.agent.max_context_chars,
                        on_phase=cb,
                    ),
                )
                self.render_fast(answer, passages)
            else:
                result = self._spinner_generate(
                    "Agent thinking",
                    lambda cb: self.comps.agent.run(question, on_step=cb),
                )
                self.render_agent_result(result)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Cancelled.[/]\n")
        except Exception as exc:
            self.console.print("\n" + self._friendly_error(exc) + "\n")

    def run(self) -> None:
        self.banner()
        self._warm_embedder()
        while True:
            try:
                question = self.console.input("[bold green]You › [/]").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye[/]")
                break
            if not question:
                continue
            if question.startswith("/"):
                if not self._handle_command(question):
                    break
                continue
            self.ask(question)
