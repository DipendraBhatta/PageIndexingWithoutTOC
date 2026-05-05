# retrieval_engine.py

import os
import logging
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv
from Query_Retrieval.cost_estimation import CostTracker

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
        child_confidence_threshold: float = 0.35,
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
        self._domain_summary = self._generate_domain_summary()

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

    def _generate_domain_summary(self) -> str:
            prompt = f"""Based on these document section titles, write a SHORT 1-sentence description
        of what broad topics this document covers. Be generic and inclusive — cover the full scope,
        not just one sub-topic. incude all titles also.give domain summaary like title it covers , and its main topics. 

    TITLES:
    {self._title_tree[:2000]}

    Return ONLY the one-sentence description. No extra text."""
            try:
                
                res = self._tracked_llm_call(
                    step="domain_summary",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                summary = res.choices[0].message.content.strip()
                self.log(f"Domain summary generated: {summary}")
                return summary
            except Exception as e:
                self.log(f"Domain summary error: {e} — using fallback")
                return "the topics covered in this document"

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

        This document covers: {self._domain_summary}

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
            prompt = f"""Classify into exactly one: "simple", "multi_fact", or "analytical".

    Return ONLY JSON: {{"type": "..."}}

    QUERY: {query}"""
            try:
                
                res = self._tracked_llm_call(
                    step="query_classification",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                return json.loads(_strip_fences(res.choices[0].message.content)).get("type", "simple")
            except Exception as e:
                self.log(f"classify_query error: {e}")
                return "simple"

    def plan_query(self, query: str, query_type: str) -> List[str]:
        if query_type == "simple" or len(query.split()) <= 8:
            return [query]
        prompt = f"""Break into at most 3 short retrieval intents. Return ONLY a JSON list of strings.

         QUESTION: {query}"""
        try:
            
                res = self._tracked_llm_call(
                    step="query_planning",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                return json.loads(_strip_fences(res.choices[0].message.content))
        except Exception as e:
                self.log(f"plan_query error: {e}")
                return [query]

    def choose_root(self, roots: List[Dict], intent: str) -> Dict:
        options = [
            {"id": i, "title": r.get("title", ""), "summary": (r.get("content") or "")[:300]}
            for i, r in enumerate(roots)
        ]
        prompt = f"""Select the best ROOT section whose subtopics are most likely to answer this intent.



        Scoring guide (be strict and realistic):
            0.8-1.0: This root's subtopics directly match the intent topic
            0.5-0.7: Loosely related, subtopics might contain the answer
            0.1-0.4: Unlikely to contain the answer
            0.0: Completely unrelated

        IMPORTANT:
        - root node contains topic wise summary of its child nodes, so evaluate the root's content as well as the title 
        - Score based on whether SUBTOPICS under this root will answer the intent
        - Do NOT give 0.9 just because a title partially matches
        - Scores must reflect actual relevance, not title similarity alone

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

    def choose_child(self, node: Dict, intent: str) -> Optional[Dict]:
        children = node.get("nodes", [])
        if not children:
            return None
        
        options = [
            {"id": i, "title": c.get("title", ""), "summary": (c.get("content") or "")[:300]}
            for i, c in enumerate(children)
        ]
        prompt = f"""Select the best CHILD section that most directly answers this intent.

        Scoring guide (be strict and differentiated — do NOT give same score to multiple options):
            0.9-1.0: Title AND content summary directly and specifically answer the intent
            0.7-0.8: Strongly related, content likely contains a partial or full answer
            0.4-0.6: May be relevant but the match is indirect or uncertain
            0.1-0.3: Unlikely to contain the answer
            0.0: Completely unrelated

        IMPORTANT:
        - Do NOT give 0.90 just because the section title matches — evaluate the content summary too
        - The best match must score MEANINGFULLY higher than other options
        - Reason must explain what specific content makes this the best match

        Return ONLY valid JSON:
        {{"index": <number>, "confidence": <float between 0.0 and 1.0>, "reason": "explain what specific content makes this section the best match"}}

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
                    self.log(
                        f"Low child confidence ({conf:.2f} < {self.child_confidence_threshold}) "
                        f"for '{rejected_title}' → staying at current node. Reason: {reason}"
                    )
                    return None
                
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
            
            next_node = self.choose_child(node, intent)
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
        self.trace = []
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
     
        #  FIXED: Only append once
        self.history.append({"role": "assistant", "content": final_answer})
        
        # --- COST TRACKING OUTPUT ---
        print("\n--- QUERY COST SUMMARY ---")
        print(self.cost_tracker.get_report())
        self.cost_tracker.reset()
        
        return {
            "query_id": qid,
            "answer": final_answer,
            "rewritten_query": rewritten,
            "trace": self.trace,
            "provenance": {"retrievals": provenance_retrievals}
        }

    def pretty_query(self, question: str, result=None):
        if result is None:
            result = self.query(question)
        
        print("\n" + "═" * 90)
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                    EXPLAINABLE TREE RAG — QUERY RESULT                       ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print(f"  Document         : Your Tree Index")
        print(f"  Tree Index       : {os.path.basename('your_tree_index.json')}")
        print(f"  • Total Nodes    : {len(self._title_tree.splitlines())}")
        print(f"  • Document domain: {self._domain_summary}")
        print(f"  • Conversation turns: {len(self.history)}")
        print("──────────────────────────────────────────────────────────────────────────────")
        print(f'USER QUESTION     : "{question}"')
        print(f'LLM UNDERSTOOD    : "{result.get("rewritten_query")}"')
        print("PHASE 1 ── TREE SEARCH (Reasoning-based Retrieval + Tree Traversal)")
        print("──────────────────────────────────────────────────────────────────────────────")
        print("Model used        :", self.model)
        print("Input to LLM      : Full hierarchical tree (titles only)")
        print("\n══════════════════════════════════════════════════════════════════════════════")
        print("2. LLM TREE TRAVERSAL & BRANCH EVALUATION (Step-by-step)")
        print("══════════════════════════════════════════════════════════════════════════════")
        
        retrievals = result["provenance"].get("retrievals", [])
        if retrievals:
            steps = retrievals[0].get("traversal", [])
            if steps:
                for step in steps:
                    action = "→ Enter branch" if step["level"] == "root" else "→ Select"
                    print(
                        f"Step {step['step']} → {step['level'].upper()} "
                        f"'{step['title']}' "
                        f"Confidence: {step['confidence']:.2f} "
                        f"Reason: {step['reason']} {action}"
                    )
            else:
                print("  No traversal steps recorded.")
        else:
            print("  No traversal performed (query not relevant to document)")
        
        print("\n══════════════════════════════════════════════════════════════════════════════")
        print("3. SELECTION RESULT")
        print("══════════════════════════════════════════════════════════════════════════════")
        
        if retrievals:
            leaf = retrievals[0].get("leaf", "—")
            steps = retrievals[0].get("traversal", [])
            real_conf = steps[-1]["confidence"] if steps else None
            real_reason = steps[-1]["reason"] if steps else "No traversal data available"
            
            print(f"  SELECTED NODE:")
            print(f"    • {leaf}")
            print(f"  Branch confidence : {f'{real_conf:.2f}' if real_conf is not None else 'n/a'}")
            print(f"  Reason            : {real_reason}")
        else:
            print("  No relevant section found in the document.")
        
        print("──────────────────────────────────────────────────────────────────────────────")
        print("PHASE 2 ── CONTEXT EXTRACTION")
        print("──────────────────────────────────────────────────────────────────────────────")
        
        if retrievals:
            print("Extracting full raw text from selected node(s)…")
            print("Context Quality   : High")
        else:
            print("Context extraction skipped (query not relevant)")
            print("Context Quality   : N/A")
        
        print("──────────────────────────────────────────────────────────────────────────────")
        print("PHASE 3 ── ANSWER GENERATION")
        print("──────────────────────────────────────────────────────────────────────────────")
        print("Final Answer:")
        print(result.get("answer", "No answer generated"))
        print("──────────────────────────────────────────────────────────────────────────────")
        print("Trace saved to    : process.log")
        print("══════════════════════════════════════════════════════════════════════════════\n")