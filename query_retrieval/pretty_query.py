
import json

class ColorSetup:
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

def _hr(char="─", color=ColorSetup.GRAY, width=W):
    print(f"{color}{char * width}{ColorSetup.RESET}")

def _header(text, bg=ColorSetup.BG_DARK, fg=ColorSetup.CYAN):
    pad = W - len(text) - 4
    print(f"{bg}{fg}{ColorSetup.BOLD}  {text}{' ' * pad}  {ColorSetup.RESET}")

def _label(key, value, key_color=ColorSetup.GRAY, val_color=ColorSetup.WHITE, indent=2):
    sp = " " * indent
    print(f"{sp}{key_color}{key:<22}{ColorSetup.RESET}{val_color}{value}{ColorSetup.RESET}")

def _conf_bar(conf: float, width: int = 20) -> str:
    filled = round(conf * width)
    empty  = width - filled
    if conf >= 0.75:   col = ColorSetup.GREEN
    elif conf >= 0.50: col = ColorSetup.YELLOW
    elif conf >= 0.35: col = ColorSetup.ORANGE
    else:              col = ColorSetup.RED
    bar = f"{col}{'' * filled}{ColorSetup.GRAY}{'░' * empty}{ColorSetup.RESET}"
    pct = f"{col}{conf:.0%}{ColorSetup.RESET}"
    return f"{bar} {pct}"

def _conf_badge(conf: float) -> str:
    if conf >= 0.75:   return f"{ColorSetup.BG_GRN}{ColorSetup.WHITE} HIGH {ColorSetup.RESET}"
    elif conf >= 0.50: return f"\033[48;5;58m{ColorSetup.WHITE} MED  {ColorSetup.RESET}"
    elif conf >= 0.35: return f"\033[48;5;130m{ColorSetup.WHITE} LOW  {ColorSetup.RESET}"
    else:              return f"{ColorSetup.BG_RED}{ColorSetup.WHITE} SKIP {ColorSetup.RESET}"

def _step_icon(level: str) -> str:
    return {"root": "◈", "child": "◆"}.get(level, "◇")

def _action_tag(action: str) -> str:
    colors = {"selected": ColorSetup.GREEN, "fallback": ColorSetup.ORANGE, "rejected": ColorSetup.RED}
    return f"{colors.get(action, ColorSetup.GRAY)}[{action.upper()}]{ColorSetup.RESET}"

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
    print(f"\n{ColorSetup.CYAN}{ColorSetup.BOLD}◈ DOCUMENT LOADED SUCCESSFULLY{ColorSetup.RESET}")
    _hr("━", ColorSetup.GRAY)
    
    # Summary Block
    print(f"  {ColorSetup.WHITE}{ColorSetup.BOLD}Domain Summary:{ColorSetup.RESET}")
    lines = _wrap(summary, W - 6)
    for line in lines:
        print(f"  {ColorSetup.ITALIC if hasattr(ColorSetup, 'ITALIC') else ''}{ColorSetup.CYAN}{line}{ColorSetup.RESET}")
    
    print()
    # Stats Row
    _label("Total Sections  :", str(total_sections), val_color=ColorSetup.GREEN)
    _label("System Status   :", "Ready for Queries", val_color=ColorSetup.GREEN)
    _hr("─", ColorSetup.GRAY)

    # Suggestions
    print(f"\n  {ColorSetup.YELLOW}🔍 Suggested Questions:{ColorSetup.RESET}")
    print(f"  {ColorSetup.DIM}• What is covered under Common Medical Events?{ColorSetup.RESET}")
    print(f"  {ColorSetup.DIM}• What services are excluded?{ColorSetup.RESET}")
    print(f"  {ColorSetup.DIM}• Explain Summary of Benefits and Coverage{ColorSetup.RESET}")
    
    print(f"\n  {ColorSetup.CYAN}Type your question below{ColorSetup.RESET} {ColorSetup.GRAY}{ColorSetup.RESET}")

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
    _hr("═", ColorSetup.CYAN)
    _header("EXPLAINABLE TREE RAG — QUERY RESULT", bg=ColorSetup.BG_DARK, fg=ColorSetup.CYAN)
    _hr("═", ColorSetup.CYAN)

    # ── DOC METADATA ─────────────────────────────────────────────────────────
    _label("Document domain :", self.get_domain_summary()[:70] + "..." 
           if len(self.get_domain_summary()) > 70 else self.get_domain_summary(), 
           val_color=ColorSetup.YELLOW)
    _label("Total sections  :", str(len(self._title_tree.splitlines())),   val_color=ColorSetup.WHITE)
    _label("Conversation    :", f"{len(self.history)} turn(s)",            val_color=ColorSetup.WHITE)
    _label("Model           :", str(self.model),                           val_color=ColorSetup.MAGENTA)

    # ── PHASE 1 : QUERY UNDERSTANDING ────────────────────────────────────────
    print()
    _hr("─", ColorSetup.BLUE)
    _header("PHASE 1 ─── QUERY UNDERSTANDING", bg=ColorSetup.BG_BLUE, fg=ColorSetup.WHITE)
    _hr("─", ColorSetup.BLUE)
    print()

    _label("User question   :", f'"{question}"', val_color=ColorSetup.WHITE)
    
    if question.strip().lower() != rewritten.strip().lower():
        _label("Rewritten       :", f'"{rewritten}"', val_color=ColorSetup.GREEN)
    else:
        _label("Rewritten       :", f"{ColorSetup.GRAY}(no change){ColorSetup.RESET}")

    print(f"\n  {ColorSetup.CYAN}Pipeline Processing:{ColorSetup.RESET}")

    # 1. Relevance Check
    rel_status = "Relevant" if relevance.get("relevant", True) else "Not Relevant"
    rel_color = ColorSetup.GREEN if relevance.get("relevant", True) else ColorSetup.RED
    _label("   • Relevance Check :", f"{rel_color}{rel_status}{ColorSetup.RESET}", val_color=ColorSetup.WHITE)
    if relevance.get("reason"):
        print(f"     {ColorSetup.DIM}→ {relevance['reason']}{ColorSetup.RESET}")

    # 2. Classification
    _label("   • Classification  :", f"{query_type.capitalize()} Query", val_color=ColorSetup.WHITE)
    if hasattr(self, 'last_classification_reason') and self.last_classification_reason:
        print(f"     {ColorSetup.DIM}→ {self.last_classification_reason}{ColorSetup.RESET}")

    # 3. Query Planning
    if len(plan) > 1:
        _label("   • Query Planning  :", f"Split into {len(plan)} intents", val_color=ColorSetup.YELLOW)
    else:
        _label("   • Query Planning  :", "Single intent", val_color=ColorSetup.GREEN)
    
    # if hasattr(self, 'last_planning_reason') and self.last_planning_reason:
    #     print(f"     {C.DIM}→ {self.last_planning_reason}{C.RESET}")

    planning_reason = getattr(self, 'last_planning_reason', None)
    
    if planning_reason:
        # Optional: Clean up very long or irrelevant reasons
        if len(planning_reason) > 180:
            planning_reason = planning_reason[:177] + "..."
        
        print(f"     {ColorSetup.DIM}→ {planning_reason}{ColorSetup.RESET}")
    else:
        print(f"     {ColorSetup.DIM}→ Single intent query (no split needed){ColorSetup.RESET}")


    print(f"  {ColorSetup.GRAY}→ Proceeding to Tree Traversal...{ColorSetup.RESET}\n")

    # ── PHASE 2 : TREE TRAVERSAL ──────────────────────────────────────────────
    print()
    _hr("─", ColorSetup.BLUE)
    _header("PHASE 2 ─── TREE TRAVERSAL  (node-by-node)", bg=ColorSetup.BG_BLUE, fg=ColorSetup.WHITE)
    _hr("─", ColorSetup.BLUE)

    if not retrievals:
        print(f"\n  {ColorSetup.RED}✗  Query deemed out-of-domain — no traversal performed.{ColorSetup.RESET}\n")
    else:
        for r_idx, retrieval in enumerate(retrievals):
            intent = retrieval.get("intent", "—")
            steps  = retrieval.get("traversal", [])
            leaf   = retrieval.get("leaf", "—")

            print()
            print(f"  {ColorSetup.CYAN}{ColorSetup.BOLD}Intent {r_idx + 1}/{len(retrievals)}{ColorSetup.RESET}"
                  f"  {ColorSetup.YELLOW}» {intent}{ColorSetup.RESET}")
            print()

            if not steps:
                print(f"    {ColorSetup.GRAY}No traversal steps recorded.{ColorSetup.RESET}")
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
                    level_col = ColorSetup.CYAN if level == "root" else ColorSetup.GREEN

                    print(f"  {ColorSetup.GRAY}{connector}{ColorSetup.RESET} "
                          f"{ColorSetup.BOLD}Step {step_num}{ColorSetup.RESET}  "
                          f"{level_col}{_step_icon(level)} {level.upper()}{ColorSetup.RESET}  "
                          f"{ColorSetup.WHITE}{ColorSetup.BOLD}{title}{ColorSetup.RESET}  "
                          f"{_conf_badge(conf)}  {_action_tag(action)}")

                    print(f"  {child_pfx}   {ColorSetup.GRAY}Confidence:{ColorSetup.RESET}  {_conf_bar(conf, width=18)}")

                    if reason:
                        lines = _wrap(reason, W - 16)
                        print(f"  {child_pfx}   {ColorSetup.GRAY}Reason:    {ColorSetup.RESET}{ColorSetup.DIM}{lines[0]}{ColorSetup.RESET}")
                        for extra in lines[1:]:
                            print(f"  {child_pfx}              {ColorSetup.DIM}{extra}{ColorSetup.RESET}")

                    if not is_last:
                        print(f"  {ColorSetup.GRAY}│{ColorSetup.RESET}")

            # Leaf summary
            print()
            last_step = steps[-1] if steps else {}
            leaf_conf = float(last_step.get("confidence", 0)) if last_step else 0
            print(f"  {ColorSetup.GRAY}▣  Landed on leaf:{ColorSetup.RESET}  "
                  f"{ColorSetup.GREEN}{ColorSetup.BOLD}{leaf}{ColorSetup.RESET}  "
                  f"{_conf_bar(leaf_conf, width=14)}")

    # ── PHASE 3 : CONTEXT EXTRACTION ─────────────────────────────────────────
    print()
    _hr("─", ColorSetup.BLUE)
    _header("PHASE 3 ─── CONTEXT EXTRACTION", bg=ColorSetup.BG_BLUE, fg=ColorSetup.WHITE)
    _hr("─", ColorSetup.BLUE)
    print()

    if retrievals:
        _label("Status          :", f"{ColorSetup.GREEN}✓  Extracted full content from selected node(s){ColorSetup.RESET}", val_color="")
        _label("Context quality :", f"{ColorSetup.GREEN}High{ColorSetup.RESET}", val_color="")
        _label("Nodes used      :", str(len(retrievals)), val_color=ColorSetup.WHITE)
    else:
        _label("Status          :", f"{ColorSetup.RED}✗  Skipped — query out of domain{ColorSetup.RESET}", val_color="")
        _label("Context quality :", f"{ColorSetup.RED}N/A{ColorSetup.RESET}", val_color="")

    # ── PHASE 4 : FINAL ANSWER ────────────────────────────────────────────────
    print()
    _hr("─", ColorSetup.BLUE)
    _header("PHASE 4 ─── FINAL ANSWER", bg=ColorSetup.BG_BLUE, fg=ColorSetup.WHITE)
    _hr("─", ColorSetup.BLUE)
    print()

    for para in answer.split("\n"):
        lines = _wrap(para, W - 4)
        if not lines:
            print()
            continue
        for line in lines:
            print(f"  {ColorSetup.WHITE}{line}{ColorSetup.RESET}")

    # ── COST SUMMARY ─────────────────────────────────────────────────────────
    print()
    _hr("─", ColorSetup.GRAY)
    _header("COST SUMMARY", bg=ColorSetup.BG_DARK, fg=ColorSetup.MAGENTA)
    _hr("─", ColorSetup.GRAY)        
    print()

    calls  = cost_report.get("calls", [])

    _label("Total LLM Calls :", str(cost_report.get("total_llm_calls", 0)),    val_color=ColorSetup.WHITE)
    _label("Total Tokens    :", f"In: {cost_report.get('total_input_tokens', 0):,}  |  "
                                f"Out: {cost_report.get('total_output_tokens', 0):,}", val_color=ColorSetup.WHITE)
    _label("Total Cost      :", f"${cost_report.get('total_cost_usd', 0):.6f} USD",    val_color=ColorSetup.YELLOW)
    _label("Total LLM Time  :", f"{cost_report.get('total_llm_time_seconds', 0):.3f}s", val_color=ColorSetup.WHITE)

    if calls:
        print()
        print(f"  {ColorSetup.GRAY}{'Step':<30} {'Model':<28} {'In':>6} {'Out':>5} {'Cost':>10} {'Time':>7}{ColorSetup.RESET}")
        print(f"  {ColorSetup.GRAY}{'─' * 90}{ColorSetup.RESET}")
        for call in calls:
            print(
                f"  {ColorSetup.DIM}"
                f"{call.get('step', ''):<30} "
                f"{call.get('model', ''):<28} "
                f"{call.get('input_tokens', 0):>6,} "
                f"{call.get('output_tokens', 0):>5,} "
                f"${call.get('cost_usd', 0):>9.6f} "
                f"{call.get('duration_seconds', 0):>6.3f}s"
                f"{ColorSetup.RESET}"
            )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    print()
    _hr("═", ColorSetup.CYAN)
    print(f"  {ColorSetup.GRAY}Trace logged  →  process.log{ColorSetup.RESET}"
          f"   {ColorSetup.GRAY}Query ID  →  {result.get('query_id', '—')}{ColorSetup.RESET}")
    _hr("═", ColorSetup.CYAN)
    print()