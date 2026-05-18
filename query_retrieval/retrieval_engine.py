# retrieval_engine.py

from datetime import datetime
import os
import logging
import json
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv
from query_retrieval.cost_estimation import CostTracker
from query_retrieval.pretty_query import ColorSetup, _header, _hr, display_welcome_info, pretty_query

load_dotenv()

def setup_logger(name: str = "ExplainableRAG") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler("process.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def _collect_all_titles(nodes, depth=0) -> List[str]:
    titles = []
    for node in (nodes if isinstance(nodes, list) else [nodes]):
        title = node.get("title", "")
        if title:
            titles.append("  " * depth + title)
        for child in node.get("nodes", []):
            titles.extend(_collect_all_titles([child], depth + 1))
    return titles

class ExplainableTreeRAG:
    def __init__(
        self,
        index_path: str,
        groq_api_key_env: str = "GROQ_API_KEY",
        llm_model_env: str = "LLM_MODEL",
        max_iterations: int = 3,
        child_confidence_threshold: float = 0.65,
    ):
        with open(index_path, "r", encoding="utf-8") as f:
         self.data = json.load(f)
        self.logger = setup_logger()
        self.client = Groq(api_key=os.getenv(groq_api_key_env))
        self.model = os.getenv(llm_model_env)
        self.trace: List[str] = []
        self.max_iterations = max_iterations
        self.child_confidence_threshold = child_confidence_threshold
        nodes = self.data if isinstance(self.data, list) else [self.data]
        self._title_tree: str = "\n".join(_collect_all_titles(nodes))
        self.history: List[Dict[str, str]] = []
        self.cost_tracker = CostTracker()
        self.traversal_steps: List[Dict] = []
        self.domain_summary = None
        
     
        display_welcome_info(self)

        ExplainableTreeRAG.pretty_query = pretty_query  



    def display_full_json_tree(self, truncate_words: int):
        """Clean, attractive, and properly formatted JSON tree display"""
        print()
        _hr("═", ColorSetup.CYAN)
        _header("FULL DOCUMENT INDEX TREE", bg=ColorSetup.BG_DARK, fg=ColorSetup.CYAN)
        _hr("═", ColorSetup.CYAN)

        def truncate_text(text: str, max_words: int) -> str:
            if not isinstance(text, str) or not text.strip():
                return text
            words = text.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "......"
            return text

        # Create a deep copy and truncate content/summary
        def clean_node(node):
            cleaned = node.copy()
            if "summary" in cleaned and cleaned["summary"]:
                cleaned["summary"] = truncate_text(cleaned["summary"], truncate_words)
            if "content" in cleaned and cleaned["content"]:
                cleaned["content"] = truncate_text(cleaned["content"], truncate_words)
            if "nodes" in cleaned:
                cleaned["nodes"] = [clean_node(child) for child in cleaned["nodes"]]
            return cleaned

        # Prepare data
        raw_data = self.data if isinstance(self.data, list) else [self.data]
        cleaned_data = [clean_node(root) for root in raw_data]

        # Pretty print JSON
        pretty_json = json.dumps(cleaned_data, indent=4, ensure_ascii=False)
        
        # Optional: Add color to keys for better readability
        colored_json = pretty_json
        colored_json = colored_json.replace('"node_id":', f'{ColorSetup.CYAN}"node_id":{ColorSetup.RESET}')
        colored_json = colored_json.replace('"title":', f'{ColorSetup.GREEN}"title":{ColorSetup.RESET}')
        colored_json = colored_json.replace('"summary":', f'{ColorSetup.YELLOW}"summary":{ColorSetup.RESET}')
        colored_json = colored_json.replace('"content":', f'{ColorSetup.MAGENTA}"content":{ColorSetup.RESET}')
        colored_json = colored_json.replace('"nodes":', f'{ColorSetup.BLUE}"nodes":{ColorSetup.RESET}')

        print(colored_json)

        _hr("═", ColorSetup.CYAN)
        print(f" {ColorSetup.GREEN}✓ Full hierarchical JSON tree displayed | "
            f"Content truncated to ~{truncate_words} words{ColorSetup.RESET}\n")

    def log(self, message: str) -> None:
        self.trace.append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} - {message}")
        self.logger.info(message)

    def _tracked_llm_call(self, step: str, messages: List[Dict], **kwargs):
        """Wrapper that tracks cost + time for every Groq call."""
        start = time.perf_counter()
        
       
        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        duration = time.perf_counter() - start
        usage = res.usage
        self.cost_tracker.log_llm_call(
            step=step,
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            duration_seconds=duration
        )
        return res

    # ─────────────────────────────────────

    def generate_and_save_domain_summary(self) -> str:
        """Generate domain summary only once"""
        if getattr(self, 'domain_summary', None):
            return self.domain_summary
                
        prompt = f"""Based on the following document section titles, write a SINGLE concise sentence 
            describing the overall domain and broad topics this document covers.

            TITLES:
            {self._title_tree[:2500]}

            Return ONLY the one-sentence summary. No extra text."""

        try:
            res = self._tracked_llm_call(
                step="domain_summary",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            summary = res.choices[0].message.content.strip().strip('"').strip("'")
            
            self.domain_summary = summary
            
           
            self._save_to_metadata({
                "domain_summary": summary,
                "total_sections": len(self._title_tree.splitlines()),
                "title_tree_length": len(self._title_tree),
                "indexed_on": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })
            
            self.logger.info(f"Domain summary saved: {summary}")
            return summary

        except Exception as e:
            self.log(f" Domain summary generation failed: {e}")
            fallback = "the topics covered in this document"
            self.domain_summary = fallback
            self._save_to_metadata({"domain_summary": fallback})
            return fallback


    def get_domain_summary(self) -> str:
        """Load from metadata first, generate only if missing"""
        if getattr(self, 'domain_summary', None):
            return self.domain_summary

        loaded = self._load_from_metadata("domain_summary")
        if loaded:
            self.domain_summary = loaded
            self.log(f" Domain summary loaded from metadata")
            return loaded

        self.log(" Domain summary not found. Generating now...")
        return self.generate_and_save_domain_summary()


    def _save_to_metadata(self, data: dict) -> bool:
        """Save metadata to JSON file"""
        try:
            if hasattr(self, 'index_path') and self.index_path:
                metadata_path = Path(self.index_path) / "document_metadata.json"
                metadata = {}

                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                    except:
                        pass

                metadata.update(data)
                metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
                
                self.log(f"Metadata saved: {list(data.keys())}")
                return True
            
            return False
        except Exception as e:
            self.log(f"Warning: Failed to save metadata: {e}")
            return False


    def _load_from_metadata(self, key: str):
        """Load from metadata file"""
        try:
            metadata_path = self._get_metadata_path()
            if not metadata_path or not metadata_path.exists():
                return None

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return metadata.get(key)
        except Exception as e:
            self.log(f"Failed to load metadata '{key}': {e}")
            return None


    def _get_metadata_path(self):
        """Helper to get metadata file path"""
        if hasattr(self, 'index_path') and self.index_path:
            return Path(self.index_path) / "document_metadata.json"
        return None


    
        # .......................................

    def _rewrite_query_with_history(self, question: str) -> str:
                if not self.history:
                    prompt = f"""Fix ONLY spelling and typo errors in this question.
        Do NOT change the meaning, topic, or intent in any way.
        Do NOT paraphrase or rephrase — only correct misspelled words.
        If the question looks correct already, return it unchanged.

        Examples of correct behavior:
            "Retauremrnt Benifits" → "Retirement Benefits"
            "helth insurence plan" → "health insurance plan"
            "What is the polcy for leve" → "What is the policy for leave"

        Return ONLY the corrected question. No explanation.

        QUESTION: {question}"""
                else:
                    history_text = "\n".join([
                        f"{turn['role'].upper()}: {turn['content']}"
                        for turn in self.history[-6:]
                    ])
                    prompt = f"""You are a query rewriter. Do exactly two things:
        1. Fix spelling/typo errors conservatively — keep the original topic and intent
        2. Resolve pronouns or vague references using conversation history

        STRICT RULES:
        - Use the History to replace pronouns (like:it, they, that, those) with the actual subjects.
        - Never change the meaning or topic of the question
        - Never guess a completely different word if correction is ambiguous — keep closest match
        "Retauremrnt" → "Retirement" (NOT "restaurants")
        - If you cannot confidently fix a typo, leave the word as-is

        Conversation History:
        {history_text}

        CURRENT USER QUESTION: {question}

        Return ONLY the corrected, self-contained question. No extra text."""
                try:
                
                    res = self._tracked_llm_call(
                        step="query_rewrite",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    
                    rewritten = res.choices[0].message.content.strip()
                    self.log(f"Query rewritten: '{question}' → '{rewritten}'")
                    return rewritten
                except Exception as e:
                    self.log(f"Query rewrite error: {e} — using original")
                    return question

    def is_query_relevant(self, query: str) -> Dict[str, Any]:
            prompt = f"""You are a topic relevance judge for a document retrieval system.

        This document covers: {self.domain_summary}

        Your ONLY job: decide if the query topic is broadly related to this document's domain.

        RULES:
        - Return relevant: true if the query could POSSIBLY be answered by a document on this topic
        - Return relevant: false ONLY if the topic is completely unrelated (e.g. asking about football scores in a medical document)
        - Do NOT judge whether the exact answer exists — only check if the topic fits the domain
        - When uncertain, always return relevant: true

        Return ONLY valid JSON: {{"relevant": true | false, "reason": "one short sentence"}}

        QUERY: {query}"""
            try:
                    
                    res = self._tracked_llm_call(
                        step="relevance_check",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    
                    out = json.loads(_strip_fences(res.choices[0].message.content))
                    self.log(f"Relevance check → {out}")
                    return out
            except Exception as e:
                    self.log(f"Relevance check error: {e} — defaulting to relevant")
                    return {"relevant": True, "reason": "proceeding (parse error)"}

    def classify_query(self, query: str) -> str:
                prompt = f"""Classify this query into exactly ONE of the following:
        - "simple"      : Straightforward, single topic question
        - "multi_fact"  : Asks about multiple different topics or comparisons at once
        - "analytical"  : Requires deep reasoning, calculation, or inference

        Return ONLY valid JSON: {{"type": "simple" | "multi_fact" | "analytical" "reason": "proper and short reason: why it is classified as such" }}

        QUERY: {query}"""

                try:
                    res = self._tracked_llm_call(
                        step="query_classification",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    
                    result = json.loads(_strip_fences(res.choices[0].message.content))
                    query_type = result.get("type", "simple").lower()
                    reason = result.get("reason", "No reason provided")
                    self.last_classification_reason = reason

                    
                    word_count = len(query.split())
                    if word_count <= 8:                   
                        query_type = "simple"
                    
                    self.log(f"Query classified as: {query_type} (words: {word_count})")
                    return query_type
                    
                except Exception as e:
                    self.last_classification_reason = str(e)
                    self.log(f"classify_query error: {e}")
                    return "simple"
                
                
    def plan_query(self, query: str, query_type: str) -> List[str]:
        """
        Decides whether to split the query into multiple retrieval intents.
        Only splits when truly beneficial.
        """
        word_count = len(query.split())

        if query_type == "simple" or word_count <= 8:
            self.log(f"plan_query: Single intent (simple query, {word_count} words)")
            return [query]

       
        prompt = f"""You are an expert at breaking down complex questions for document retrieval.

            Analyze the query and decide:
            - If it has **multiple distinct topics**, break it into **at most 3  not note more than 3 but you can break less than 3** clear, meaningful retrieval intents.
            - If it is mostly one topic (even if complex), return it as a single intent.

            Rules for good splitting:
            - Only split if the topics are different enough that they would likely require separate retrievals to answer well.
            - Each intent must be self-contained and focused.
            - Do NOT split if the sub-parts are too similar or tightly related.
            - Return at most 3 intents.

            Return ONLY a valid JSON object in this exact format:

                {{
                "intents": ["first clear intent", "second clear intent", ...],
                "reason": "short and clear explanation of your decision, why do you think these intents are distinct and beneficial for retrieval?"
                }}

        QUERY: {query}"""

        try:
            res = self._tracked_llm_call(
                step="query_planning",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            data = json.loads(_strip_fences(res.choices[0].message.content))
            
            plan = data.get("intents", [query])
            reason = data.get("reason", "No reason provided")

            if not isinstance(plan, list):
                plan = [query]

            
            plan = plan[:3]

            self.last_planning_reason = reason
            self.log(f"Planning: {len(plan)} intent(s) | Reason: {reason}")

            return plan

        except Exception as e:
            self.last_planning_reason = f"Error in planning: {str(e)}"
            self.log(f"plan_query error: {e}")
            return [query]
            
            




    def _robust_json_parse(text: str) -> Dict:
        """
        Extracts and parses the first JSON object found in the text.
        Handles Markdown fences and conversational filler.
        """
        try:
            # Regex to find everything between the first '{' and the last '}'
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            return json.loads(text) # Fallback to standard
        except Exception as e:
            raise ValueError(f"Regex JSON parse failed: {e}")

    def choose_root(self, roots: List[Dict], intent: str) -> Dict:


        options = []
        for i, r in enumerate(roots):
            # Fallback through potential keys where your summary might live
            summary_text = r.get("summary") or r.get("content") or ""
            # Remove redundant "title:" or "summary:" labels inside the string
            clean_summary = summary_text.replace("title:", "").replace("summary:", "").strip()
            
            options.append({
                "id": i, 
                "title": r.get("title", ""), 
                "summary": clean_summary[:1500] 
            })
        prompt = f"""Select the best ROOT section whose subtopics are most likely to answer this intent.



        Scoring guide (be strict and realistic):
            0.9–1.0: Title AND summary directly and specifically answer the intent
            0.7–0.8: Strongly related, content likely contains a partial or full answer
            0.4–0.6: Indirect or uncertain relevance
            0.1–0.3: Very unlikely
            0.0: Completely unrelated

        IMPORTANT:
            - The root node already contains its child node titles and summaries. Each child summary explains what that child section is about.
            - Read both the child title and its summary carefully before scoring.
            - If INTENT mentions "child", "children", "pediatric", "eye care", or "dental", prioritize options whose title or summary explicitly mention those terms.
            - Do not give a high score just because the title looks similar, the summary must confirm relevance.
            - Better check the content of the child summaries to see if they mention keywords related to the intent, rather than just relying on the root title.
            - The best match must score meaningfully higher than other options.
            - Your reason must reference specific content in the child summary that matches the query.


        Return ONLY valid JSON:
        {{"index": <number>, "confidence": <float between 0.0 and 1.0>, "reason": "explain why this root's subtopics likely answer the intent"}}

        OPTIONS:
        {json.dumps(options, indent=2)}

        INTENT: {intent}"""
        try:
            
                res = self._tracked_llm_call(
                    step="choose_root",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                out = json.loads(_strip_fences(res.choices[0].message.content))
                idx = out.get("index", 0)
                conf = float(out.get("confidence", 0.5))
                reason = out.get("reason", "No reason provided")
                idx = max(0, min(idx, len(roots) - 1))
                
                self.traversal_steps.append({
                    "step": len(self.traversal_steps) + 1,
                    "level": "root",
                    "title": roots[idx].get("title", "Unknown"),
                    "confidence": round(conf, 2),
                    "reason": reason,
                    "action": "selected"
                })
                
                self.log(f"Root selected: '{roots[idx].get('title')}' conf={conf:.2f} reason={reason}")
                return roots[idx]
        except Exception as e:
                self.log(f"choose_root error: {e} — falling back to first root")
                if roots:
                    self.traversal_steps.append({
                        "step": len(self.traversal_steps) + 1,
                        "level": "root",
                        "title": roots[0].get("title", "Unknown"),
                        "confidence": 0.0,
                        "reason": f"Fallback due to error: {e}",
                        "action": "fallback"
                    })
                return roots[0] if roots else {}



    def choose_child(self, node: Dict, intent: str, parent_reason: str = "") -> Optional[Dict]:
            children = node.get("nodes", [])
            if not children:
                return None

            options = [
                {"id": i, "title": c.get("title", ""), "summary": (c.get("content") or "")[:4000]}
                for i, c in enumerate(children)
            ]

            hint_titles = []
            strong_hint = ""
            if parent_reason:
                parent_lower = parent_reason.lower()
                for opt in options:
                    title_lower = opt["title"].strip().lower()
                    if title_lower and title_lower in parent_lower:
                        hint_titles.append(opt["title"])
                
                if hint_titles:
                    strong_hint = f"""
            CRITICAL PARENT GUIDANCE (HIGH PRIORITY):
            The ROOT selection reasoning specifically mentioned these sections as relevant:
            {hint_titles}
            
            You MUST give strong preference to any of these sections if their content also supports the intent.
            This is the most important signal — do not ignore it.
            """

            hint_block = ""
            if hint_titles:
                hint_block = f"""
            PARENT CONTEXT HINT (use this to break ties):
            The parent selection reasoning already identified these sections as likely relevant:
            {hint_titles}
            If any of these appear in the OPTIONS below, strongly prefer them over others.
            This is NOT a hardcoded rule — only apply it if the content also supports it.
            """

            prompt = f"""Select the best CHILD section that most directly answers this intent.

                Scoring guide (be strict and differentiated — do NOT give same score to multiple options):
                0.9-1.0: Title AND content summary, multiple keywords directly and specifically answer the intent
                0.7-0.8: Strongly related, keywords in content likely contains a partial or full answer
                0.4-0.6: May be relevant but the match is indirect or uncertain
                0.1-0.3: Unlikely to contain the answer
                0.0: Completely unrelated

                IMPORTANT:
                - If INTENT mentions "child", "children", "pediatric", "eye care", or "dental", prioritize options whose title or summary explicitly mention those terms.
                - Do NOT confuse the words like child, children — they reflect the same meaning. Read carefully.
                - Do NOT give 0.90 just because the section title matches — evaluate the content summary too.
                - The best match must score MEANINGFULLY higher than other options.
                - Reason must explain what specific content makes this the best match.
                - A section that says "does not directly answer" in its own reasoning must score below 0.60.
                {strong_hint}
                {hint_block}

                Return ONLY valid JSON:
                {{"index": <number>, "confidence": <float between 0.0 and 1.0>, "reason": "explain what specific content makes this the best match"}}

                OPTIONS:
                {json.dumps(options, indent=2)}

                INTENT: {intent}"""

            try:
                res = self._tracked_llm_call(
                    step="choose_child",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                out = json.loads(_strip_fences(res.choices[0].message.content))
                idx = out.get("index", -1)
                conf = float(out.get("confidence", 0.0))
                reason = out.get("reason", "No reason provided")

                if idx < 0 or idx >= len(children) or conf < self.child_confidence_threshold:
                    rejected_title = children[idx].get("title", "N/A") if 0 <= idx < len(children) else "N/A"
                    self.log(f"Low child confidence ({conf:.2f}) for '{rejected_title}' → staying at current node.")
                    return None

                # Hedging check
                HEDGE_PHRASES = [
                    "does not directly answer",
                    "does not explicitly mention",
                    "does not directly address",
                    "by extension",
                    "possibly",
                    "implies that there might",
                    "although it does not",
                    "may include",
                    "not explicitly",
                ]
                reason_lower = reason.lower()
                is_hedging = any(phrase in reason_lower for phrase in HEDGE_PHRASES)

                if is_hedging and conf < 0.75:
                    self.log(
                        f"Hedging language detected in reason for '{children[idx].get('title')}' "
                        f"(conf={conf:.2f}) → rejecting. Reason: {reason}"
                    )
                    # Strong fallback to hinted node
                    if hint_titles:
                        for child in children:
                            if child.get("title", "") in hint_titles:
                                self.log(f"Falling back to hinted node: '{child.get('title')}'")
                                self.traversal_steps.append({
                                    "step": len(self.traversal_steps) + 1,
                                    "level": "child",
                                    "title": child.get("title", "Unknown"),
                                    "confidence": round(conf, 2),
                                    "reason": f"[Strong Hint Fallback] {reason}",
                                    "action": "selected"
                                })
                                return child
                    return None

                # Record successful selection
                self.traversal_steps.append({
                    "step": len(self.traversal_steps) + 1,
                    "level": "child",
                    "title": children[idx].get("title", "Unknown"),
                    "confidence": round(conf, 2),
                    "reason": reason,
                    "action": "selected"
                })

                self.log(f"Child selected: '{children[idx].get('title')}' conf={conf:.2f} reason={reason}")
                return children[idx]

            except Exception as e:
                self.log(f"choose_child error: {e}")
                return None
        


    

    def extract_answer(self, node: Dict, intent: str, context_trail: List[str]) -> str:
        parts = []
        if context_trail:
            parts.append("=== ANCESTOR CONTEXT ===")
            parts.append("\n".join(context_trail))
        
        content = (node.get("content") or "").strip()
        if content:
            parts.append(f"=== SECTION: {node.get('title', '')} ===")
            parts.append(content)
                
            full_context = "\n\n".join(parts)
            prompt = f"""Answer using ONLY the CONTEXT below. Be specific and concise.
        If the answer is not present in the context, say "Not found in document".

        CONTEXT:
        {full_context}

        QUESTION:
        {intent}"""
        try:
           
            res = self._tracked_llm_call(
                step="extract_answer",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            return res.choices[0].message.content.strip()
        except Exception as e:
            self.log(f"extract_answer error: {e}")
            return "Not found in document"

    

    def retrieve_for_intent(self, intent: str) -> Dict[str, Any]:
            start = time.time()
            self.traversal_steps = []
            nodes = self.data if isinstance(self.data, list) else [self.data]
            node = self.choose_root(nodes, intent)
            context_trail: List[str] = []
            depth = 0

           
            root_reason = self.traversal_steps[-1]["reason"] if self.traversal_steps else ""

            while True:
                title = node.get("title", "Unknown")
                snippet = (node.get("content") or "")[:200]
                context_trail.append(f"[{title}]: {snippet}")

                children = node.get("nodes", [])
                if not children:
                    ans = self.extract_answer(node, intent, context_trail[:-1])
                    return {
                        "answer": ans,
                        "leaf": title,
                        "elapsed": time.time() - start,
                        "traversal": self.traversal_steps
                    }

                next_node = self.choose_child(node, intent, parent_reason=root_reason)

                if not next_node:
                    ans = self.extract_answer(node, intent, context_trail[:-1])
                    return {
                        "answer": ans,
                        "leaf": title,
                        "elapsed": time.time() - start,
                        "traversal": self.traversal_steps
                    }

                node = next_node
                depth += 1




    def synthesize_answer(self, original_question: str, fragments: List[Dict[str, str]], query_type: str) -> str:
            
            if not fragments:
                    return "The answer was not found in the document."
                
            if query_type == "simple" and len(fragments) == 1:
                    return fragments[0]["answer"]
                
            retrieved_facts = "\n\n".join(
                    f"[Retrieved for: {f['intent']}]\n{f['answer']}" for f in fragments
                )
            prompt = f"""Using the RETRIEVED FACTS below, answer the ORIGINAL QUESTION clearly and concisely.

                RETRIEVED FACTS:
                {retrieved_facts}

                ORIGINAL QUESTION: {original_question}"""
            try:
                    
                    res = self._tracked_llm_call(
                        step="synthesize_answer",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    
                    return res.choices[0].message.content.strip()
            except Exception as e:
                    self.log(f"synthesize_answer error: {e}")
                    return "\n\n".join(f["answer"] for f in fragments)


    def query(self, question: str, stream: bool = False) -> Dict[str, Any]:

        self.last_planning_reason = None
        self.last_classification_reason = None
        self.last_relevance_reason = None  

        qid = str(uuid.uuid4())
        self.history.append({"role": "user", "content": question})
        
        rewritten = self._rewrite_query_with_history(question)
        
        relevance = self.is_query_relevant(rewritten)
        if not relevance.get("relevant", True):
            msg = (
                f"Your question does not appear to be related to the documents in this knowledge base.\n"
                f"Reason: {relevance.get('reason', 'Out of domain')}"
            )
            self.history.append({"role": "assistant", "content": msg})
            return {
                "query_id": qid,
                "answer": msg,
                "rewritten_query": rewritten,
                "trace": self.trace,
                "provenance": {"retrievals": []}
            }
        
        query_type = self.classify_query(rewritten)
        plan = self.plan_query(rewritten, query_type)
        
        provenance_retrievals: List[Dict] = []
        useful_fragments: List[Dict[str, str]] = []
        
        for intent in plan[:self.max_iterations]:
            res = self.retrieve_for_intent(intent)
            provenance_retrievals.append({
                "intent": intent,
                "leaf": res.get("leaf"),
                "traversal": res.get("traversal", [])
            })
            answer = (res.get("answer") or "").strip()
            if "Not found" not in answer:
                useful_fragments.append({"intent": intent, "answer": answer})
        
        final_answer = self.synthesize_answer(question, useful_fragments, query_type)
     
      
        self.history.append({"role": "assistant", "content": final_answer})

        cost_report = self.cost_tracker.get_report()
      

        self.cost_tracker.reset()
        
        return {
            "query_id": qid,
            "answer": final_answer,
            "rewritten_query": rewritten,
            "trace": self.trace,
            "provenance": {"retrievals": provenance_retrievals},
            "cost_report": cost_report,
            "relevance": relevance,
            "query_type": query_type,
            "plan": plan,
            "classification_reason": getattr(self, 'last_classification_reason', None),
            "planning_reason": getattr(self, 'last_planning_reason', None)
        }
        

  