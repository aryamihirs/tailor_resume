# agents/base_react_agent.py

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from typing import List, Any

class BaseReActAgent:
    def __init__(self, name: str, description: str, system_prompt: str):
        self.name = name
        self.description = description
        self.llm = OpenAI(model="gpt-3.5-turbo")  # Change to "gpt-4" if needed
        self.tools = self.get_tools()
        self.agent = ReActAgent.from_tools(self.tools, llm=self.llm, verbose=True)
        self.update_system_prompt(system_prompt)

    def get_tools(self) -> List[FunctionTool]:
        return []

    def execute_task(self, task: str) -> Any:
        return self.agent.chat(task)

    def update_system_prompt(self, new_prompt: str):
        self.agent.update_prompts({"agent_worker:system_prompt": new_prompt})
