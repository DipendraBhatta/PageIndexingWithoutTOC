# cost_estimation.py

import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class LLMCallRecord:
    step: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_seconds: float

class CostTracker:
    def __init__(self, pricing_config: Optional[Dict[str, Dict[str, float]]] = None):
        # Latest Groq pricing (April 2026) — per 1M tokens
        # Source: https://groq.com/pricing
        self.pricing = pricing_config or {
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "llama3-8b-8192": {"input": 0.05, "output": 0.08},
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama3-70b-8192": {"input": 0.59, "output": 0.79},
            "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
            "gpt-oss-20b": {"input": 0.075, "output": 0.30},
            "gpt-oss-120b": {"input": 0.15, "output": 0.60},
            "gemma2-9b-it": {"input": 0.20, "output": 0.20},
            "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
            # Add any new Groq model here (model_id as key)
        }
        self.calls: List[LLMCallRecord] = []
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a single LLM call based on token usage."""
        if model not in self.pricing:
            print(f" Warning: No pricing found for model '{model}'. Cost set to 0.")
            return 0.0
        
        p = self.pricing[model]
        # Cost = (input_tokens * input_price + output_tokens * output_price) / 1M
        return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    
    def log_llm_call(
        self,
        step: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float
    ) -> None:
        """Log a single LLM API call with its cost and timing."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        record = LLMCallRecord(
            step=step,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_seconds=duration_seconds
        )
        self.calls.append(record)
        
        # This is useful for debugging, but can be commented out in production
        print(f" TRACKED → {step:28} | {model:25} | ${cost:.6f} | {duration_seconds:.3f}s")
    
    def get_report(self, as_json: bool = False) -> Dict:
        """Generate a comprehensive cost and timing report."""
        if not self.calls:
            print("No LLM calls recorded yet.")
            return {}
        
        total_calls = len(self.calls)
        total_cost = sum(call.cost_usd for call in self.calls)
        total_time = sum(call.duration_seconds for call in self.calls)

        total_input_tokens = sum(call.input_tokens for call in self.calls)
        total_output_tokens = sum(call.output_tokens for call in self.calls)
        
        print("\n" + "="*80)
        print("           EXPLAINABLE TREE RAG — COST & TIME TRACKER REPORT")
        print("="*80)
        print(f"  Total LLM Calls       : {total_calls}")
        print(f"  Total Input Tokens    : {total_input_tokens:,}")
        print(f"  Total Output Tokens   : {total_output_tokens:,}")
        print(f"  Total Cost            : ${total_cost:.6f} USD")
        print(f"  Total LLM Time (sum)  : {total_time:.3f} seconds")
        print("\n  Detailed Breakdown:")
        print("-" * 90)
        
        for call in self.calls:
            print(
                f"  • {call.step:28} | "
                f"{call.model:25} | "
                f"In: {call.input_tokens:6,} | "
                f"Out: {call.output_tokens:6,} | "
                f"${call.cost_usd:.6f} | "
                f"{call.duration_seconds:.3f}s"
            )
        
        print("="*80)
        
        # Prepare structured report
        report = {
            "total_llm_calls": total_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_llm_time_seconds": round(total_time, 3),
            "calls": [
                {
                    "step": c.step,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                    "duration_seconds": round(c.duration_seconds, 3)
                }
                for c in self.calls
            ]
        }
        
        if as_json:
            print("\n JSON for logging / dashboard:")
            print(json.dumps(report, indent=2))
        
        return report
    
    def reset(self):
        """
        Clear all tracked calls.
        Call this after each query if you process many queries in a loop.
        """
        self.calls.clear()