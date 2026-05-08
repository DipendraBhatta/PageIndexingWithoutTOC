
import json
# ── ANSI palette ─────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    ORANGE  = "\033[38;5;214m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    GRAY    = "\033[90m"
    BG_DARK = "\033[48;5;235m"
    BG_BLUE = "\033[48;5;17m"
    BG_GRN  = "\033[48;5;22m"
    BG_RED  = "\033[48;5;52m"

W = 90  # total console width

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hr(char="─", color=C.GRAY, width=W):
    print(f"{color}{char * width}{C.RESET}")

def _header(text, bg=C.BG_DARK, fg=C.CYAN):
    pad = W - len(text) - 4
    print(f"{bg}{fg}{C.BOLD}  {text}{' ' * pad}  {C.RESET}")

def _label(key, value, key_color=C.GRAY, val_color=C.WHITE, indent=2):
    sp = " " * indent
    print(f"{sp}{key_color}{key:<22}{C.RESET}{val_color}{value}{C.RESET}")

def _conf_bar(conf: float, width: int = 20) -> str:
    filled = round(conf * width)
    empty  = width - filled
    if conf >= 0.75:   col = C.GREEN
    elif conf >= 0.50: col = C.YELLOW
    elif conf >= 0.35: col = C.ORANGE
    else:              col = C.RED
    bar = f"{col}{'' * filled}{C.GRAY}{'░' * empty}{C.RESET}"
    pct = f"{col}{conf:.0%}{C.RESET}"
    return f"{bar} {pct}"

def _conf_badge(conf: float) -> str:
    if conf >= 0.75:   return f"{C.BG_GRN}{C.WHITE} HIGH {C.RESET}"
    elif conf >= 0.50: return f"\033[48;5;58m{C.WHITE} MED  {C.RESET}"
    elif conf >= 0.35: return f"\033[48;5;130m{C.WHITE} LOW  {C.RESET}"
    else:              return f"{C.BG_RED}{C.WHITE} SKIP {C.RESET}"

def _step_icon(level: str) -> str:
    return {"root": "◈", "child": "◆"}.get(level, "◇")

def _action_tag(action: str) -> str:
    colors = {"selected": C.GREEN, "fallback": C.ORANGE, "rejected": C.RED}
    return f"{colors.get(action, C.GRAY)}[{action.upper()}]{C.RESET}"

def _wrap(text: str, max_w: int):
    """Word-wrap text into a list of lines, each under max_w chars."""
    words = text.split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) > max_w:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def display_welcome_info(self):
    """Clean, single-border welcome message"""
    summary = self.get_domain_summary()
    total_sections = len(self._title_tree.splitlines()) if hasattr(self, '_title_tree') else 0
    
    # Header
    print(f"\n{C.CYAN}{C.BOLD}◈ DOCUMENT LOADED SUCCESSFULLY{C.RESET}")
    _hr("━", C.GRAY)
    
    # Summary Block
    print(f"  {C.WHITE}{C.BOLD}Domain Summary:{C.RESET}")
    lines = _wrap(summary, W - 6)
    for line in lines:
        print(f"  {C.ITALIC if hasattr(C, 'ITALIC') else ''}{C.CYAN}{line}{C.RESET}")
    
    print()
    # Stats Row
    _label("Total Sections  :", str(total_sections), val_color=C.GREEN)
    _label("System Status   :", "Ready for Queries", val_color=C.GREEN)
    _hr("─", C.GRAY)

    # Suggestions
    print(f"\n  {C.YELLOW}🔍 Suggested Questions:{C.RESET}")
    print(f"  {C.DIM}• What is covered under Common Medical Events?{C.RESET}")
    print(f"  {C.DIM}• What services are excluded?{C.RESET}")
    print(f"  {C.DIM}• Explain Summary of Benefits and Coverage{C.RESET}")
    
    print(f"\n  {C.CYAN}Type your question below{C.RESET} {C.GRAY}{C.RESET}")

def pretty_query(self, question: str, result=None):

    if result is None:
        result = self.query(question)

    rewritten     = result.get("rewritten_query", question)
    answer        = result.get("answer", "No answer generated.")
    retrievals    = result.get("provenance", {}).get("retrievals", [])
    relevance     = result.get("relevance", {})
    query_type    = result.get("query_type", "simple")
    plan          = result.get("plan", [rewritten])
    cost_report   = result.get("cost_report") or self.cost_tracker.get_report()

    # ── TOP BANNER ────────────────────────────────────────────────────────────
    print()
    _hr("═", C.CYAN)
    _header("EXPLAINABLE TREE RAG — QUERY RESULT", bg=C.BG_DARK, fg=C.CYAN)
    _hr("═", C.CYAN)

    # ── DOC METADATA ─────────────────────────────────────────────────────────
    _label("Document domain :", self.get_domain_summary()[:70] + "..." 
           if len(self.get_domain_summary()) > 70 else self.get_domain_summary(), 
           val_color=C.YELLOW)
    _label("Total sections  :", str(len(self._title_tree.splitlines())),   val_color=C.WHITE)
    _label("Conversation    :", f"{len(self.history)} turn(s)",            val_color=C.WHITE)
    _label("Model           :", str(self.model),                           val_color=C.MAGENTA)

    # ── PHASE 1 : QUERY UNDERSTANDING ────────────────────────────────────────
    print()
    _hr("─", C.BLUE)
    _header("PHASE 1 ─── QUERY UNDERSTANDING", bg=C.BG_BLUE, fg=C.WHITE)
    _hr("─", C.BLUE)
    print()

    _label("User question   :", f'"{question}"', val_color=C.WHITE)
    
    if question.strip().lower() != rewritten.strip().lower():
        _label("Rewritten       :", f'"{rewritten}"', val_color=C.GREEN)
    else:
        _label("Rewritten       :", f"{C.GRAY}(no change){C.RESET}")

    print(f"\n  {C.CYAN}Pipeline Processing:{C.RESET}")

    # 1. Relevance Check
    rel_status = "Relevant" if relevance.get("relevant", True) else "Not Relevant"
    rel_color = C.GREEN if relevance.get("relevant", True) else C.RED
    _label("   • Relevance Check :", f"{rel_color}{rel_status}{C.RESET}", val_color=C.WHITE)
    if relevance.get("reason"):
        print(f"     {C.DIM}→ {relevance['reason']}{C.RESET}")

    # 2. Classification
    _label("   • Classification  :", f"{query_type.capitalize()} Query", val_color=C.WHITE)
    if hasattr(self, 'last_classification_reason') and self.last_classification_reason:
        print(f"     {C.DIM}→ {self.last_classification_reason}{C.RESET}")

    # 3. Query Planning
    if len(plan) > 1:
        _label("   • Query Planning  :", f"Split into {len(plan)} intents", val_color=C.YELLOW)
    else:
        _label("   • Query Planning  :", "Single intent", val_color=C.GREEN)
    
    # if hasattr(self, 'last_planning_reason') and self.last_planning_reason:
    #     print(f"     {C.DIM}→ {self.last_planning_reason}{C.RESET}")

    planning_reason = getattr(self, 'last_planning_reason', None)
    
    if planning_reason:
        # Optional: Clean up very long or irrelevant reasons
        if len(planning_reason) > 180:
            planning_reason = planning_reason[:177] + "..."
        
        print(f"     {C.DIM}→ {planning_reason}{C.RESET}")
    else:
        print(f"     {C.DIM}→ Single intent query (no split needed){C.RESET}")


    print(f"  {C.GRAY}→ Proceeding to Tree Traversal...{C.RESET}\n")

    # ── PHASE 2 : TREE TRAVERSAL ──────────────────────────────────────────────
    print()
    _hr("─", C.BLUE)
    _header("PHASE 2 ─── TREE TRAVERSAL  (node-by-node)", bg=C.BG_BLUE, fg=C.WHITE)
    _hr("─", C.BLUE)

    if not retrievals:
        print(f"\n  {C.RED}✗  Query deemed out-of-domain — no traversal performed.{C.RESET}\n")
    else:
        for r_idx, retrieval in enumerate(retrievals):
            intent = retrieval.get("intent", "—")
            steps  = retrieval.get("traversal", [])
            leaf   = retrieval.get("leaf", "—")

            print()
            print(f"  {C.CYAN}{C.BOLD}Intent {r_idx + 1}/{len(retrievals)}{C.RESET}"
                  f"  {C.YELLOW}» {intent}{C.RESET}")
            print()

            if not steps:
                print(f"    {C.GRAY}No traversal steps recorded.{C.RESET}")
            else:
                for i, step in enumerate(steps):
                    is_last   = (i == len(steps) - 1)
                    level     = step.get("level", "?")
                    title     = step.get("title", "Unknown")
                    conf      = float(step.get("confidence", 0))
                    reason    = step.get("reason", "")
                    action    = step.get("action", "selected")
                    step_num  = step.get("step", i + 1)

                    connector = "└──" if is_last else "├──"
                    child_pfx = "    " if is_last else "│   "
                    level_col = C.CYAN if level == "root" else C.GREEN

                    print(f"  {C.GRAY}{connector}{C.RESET} "
                          f"{C.BOLD}Step {step_num}{C.RESET}  "
                          f"{level_col}{_step_icon(level)} {level.upper()}{C.RESET}  "
                          f"{C.WHITE}{C.BOLD}{title}{C.RESET}  "
                          f"{_conf_badge(conf)}  {_action_tag(action)}")

                    print(f"  {child_pfx}   {C.GRAY}Confidence:{C.RESET}  {_conf_bar(conf, width=18)}")

                    if reason:
                        lines = _wrap(reason, W - 16)
                        print(f"  {child_pfx}   {C.GRAY}Reason:    {C.RESET}{C.DIM}{lines[0]}{C.RESET}")
                        for extra in lines[1:]:
                            print(f"  {child_pfx}              {C.DIM}{extra}{C.RESET}")

                    if not is_last:
                        print(f"  {C.GRAY}│{C.RESET}")

            # Leaf summary
            print()
            last_step = steps[-1] if steps else {}
            leaf_conf = float(last_step.get("confidence", 0)) if last_step else 0
            print(f"  {C.GRAY}▣  Landed on leaf:{C.RESET}  "
                  f"{C.GREEN}{C.BOLD}{leaf}{C.RESET}  "
                  f"{_conf_bar(leaf_conf, width=14)}")

    # ── PHASE 3 : CONTEXT EXTRACTION ─────────────────────────────────────────
    print()
    _hr("─", C.BLUE)
    _header("PHASE 3 ─── CONTEXT EXTRACTION", bg=C.BG_BLUE, fg=C.WHITE)
    _hr("─", C.BLUE)
    print()

    if retrievals:
        _label("Status          :", f"{C.GREEN}✓  Extracted full content from selected node(s){C.RESET}", val_color="")
        _label("Context quality :", f"{C.GREEN}High{C.RESET}", val_color="")
        _label("Nodes used      :", str(len(retrievals)), val_color=C.WHITE)
    else:
        _label("Status          :", f"{C.RED}✗  Skipped — query out of domain{C.RESET}", val_color="")
        _label("Context quality :", f"{C.RED}N/A{C.RESET}", val_color="")

    # ── PHASE 4 : FINAL ANSWER ────────────────────────────────────────────────
    print()
    _hr("─", C.BLUE)
    _header("PHASE 4 ─── FINAL ANSWER", bg=C.BG_BLUE, fg=C.WHITE)
    _hr("─", C.BLUE)
    print()

    for para in answer.split("\n"):
        lines = _wrap(para, W - 4)
        if not lines:
            print()
            continue
        for line in lines:
            print(f"  {C.WHITE}{line}{C.RESET}")

    # ── COST SUMMARY ─────────────────────────────────────────────────────────
    print()
    _hr("─", C.GRAY)
    _header("COST SUMMARY", bg=C.BG_DARK, fg=C.MAGENTA)
    _hr("─", C.GRAY)
    print()

    calls  = cost_report.get("calls", [])

    _label("Total LLM Calls :", str(cost_report.get("total_llm_calls", 0)),    val_color=C.WHITE)
    _label("Total Tokens    :", f"In: {cost_report.get('total_input_tokens', 0):,}  |  "
                                f"Out: {cost_report.get('total_output_tokens', 0):,}", val_color=C.WHITE)
    _label("Total Cost      :", f"${cost_report.get('total_cost_usd', 0):.6f} USD",    val_color=C.YELLOW)
    _label("Total LLM Time  :", f"{cost_report.get('total_llm_time_seconds', 0):.3f}s", val_color=C.WHITE)

    if calls:
        print()
        print(f"  {C.GRAY}{'Step':<30} {'Model':<28} {'In':>6} {'Out':>5} {'Cost':>10} {'Time':>7}{C.RESET}")
        print(f"  {C.GRAY}{'─' * 90}{C.RESET}")
        for call in calls:
            print(
                f"  {C.DIM}"
                f"{call.get('step', ''):<30} "
                f"{call.get('model', ''):<28} "
                f"{call.get('input_tokens', 0):>6,} "
                f"{call.get('output_tokens', 0):>5,} "
                f"${call.get('cost_usd', 0):>9.6f} "
                f"{call.get('duration_seconds', 0):>6.3f}s"
                f"{C.RESET}"
            )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    print()
    _hr("═", C.CYAN)
    print(f"  {C.GRAY}Trace logged  →  process.log{C.RESET}"
          f"   {C.GRAY}Query ID  →  {result.get('query_id', '—')}{C.RESET}")
    _hr("═", C.CYAN)
    print()