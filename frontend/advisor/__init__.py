"""Wheat field advisor: photos + field details in, a grounded condition report out.

Pipeline (LangGraph): intake -> detect photos -> locate -> weather + soil -> find gaps -> analyze
-> write report (local Ollama model) -> validate -> report (or rule-based fallback).
"""
