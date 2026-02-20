# VisionClick Agent

A local, vision-driven UI automation agent that controls a desktop browser using screenshots and natural language — no browser extensions or DOM access needed. A vision model (`qwen3-vl:30b`) interprets the screen and locates elements; a reasoning LLM (`gpt-oss`) plans and executes the interaction sequence step by step.

**Demo:** https://www.youtube.com/watch?v=lLDhVaE1FPk

---

## Setup

```sh
ollama pull qwen3-vl:30b
ollama pull gpt-oss:latest
pip install -e .
python src/agent.py
```