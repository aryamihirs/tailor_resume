# agents/job_description_keyword_extraction.py

from .base_react_agent import BaseReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict

class JobDescriptionKeywordExtractionAgent(BaseReActAgent):
    def __init__(self):
        system_prompt = """
        You are an expert in analyzing job descriptions and extracting key skills, qualifications, and requirements. 
        Your task is to identify the most important keywords that represent the core competencies and qualifications sought in the job description.
        Focus on technical skills, soft skills, educational requirements, and any specific experiences mentioned.
        Provide your output as a list of keywords, categorized and ranked by importance.
        """
        super().__init__("JobKeywordExtraction", "Extract relevant keywords from the job description", system_prompt)

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.extract_keywords,
                name="extract_keywords",
                description="Extract key skills and requirements from a job description"
            )
        ]

    def extract_keywords(self, job_description: str) -> Dict[str, List[Dict[str, any]]]:
        task = f"Analyze this job description and extract the key skills, qualifications, and requirements: {job_description}"
        result = self.execute_task(task)
        return self.parse_result(result.response)

    def parse_result(self, result: str) -> Dict[str, List[Dict[str, any]]]:
        lines = result.strip().split('\n')
        parsed_result = {}
        current_category = None

        for line in lines:
            if ':' in line and not line.strip().endswith(':'):
                category, content = line.split(':', 1)
                current_category = category.strip()
                if current_category not in parsed_result:
                    parsed_result[current_category] = []
                keyword, importance = content.rsplit('(', 1)
                keyword = keyword.strip()
                importance = int(importance.rstrip(')').strip())
                parsed_result[current_category].append({"keyword": keyword, "importance": importance})
            elif current_category and line.strip():
                keyword, importance = line.rsplit('(', 1)
                keyword = keyword.strip()
                importance = int(importance.rstrip(')').strip())
                parsed_result[current_category].append({"keyword": keyword, "importance": importance})

        return parsed_result
