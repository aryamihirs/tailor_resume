# agents/resume_keyword_extraction.py

from .base_react_agent import BaseReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class ResumeKeywordExtractionAgent(BaseReActAgent):
    def __init__(self):
        system_prompt = """
        You are an expert in analyzing resumes and extracting key skills, experiences, and qualifications. 
        Your task is to identify the most important keywords that represent the core competencies and 
        qualifications presented in the resume.

        Consider the following aspects when extracting keywords:
        1. Technical skills and tools
        2. Soft skills and personal qualities
        3. Industry-specific terminology
        4. Job titles and roles
        5. Educational qualifications and certifications
        6. Notable achievements and metrics
        7. Project names or types
        8. Company names or types (e.g., Fortune 500, startup)

        Provide a comprehensive list of keywords, categorized by type (e.g., technical skills, soft skills, etc.). 
        Also, assign a relevance score to each keyword based on its prominence and importance in the resume.
        """
        super().__init__("ResumeKeywordExtraction", "Extract relevant keywords from a resume", system_prompt)

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.extract_keywords_from_text,
                name="extract_keywords",
                description="Extract key skills and qualifications from a resume"
            )
        ]

    def extract_keywords_from_text(self, text: str) -> List[str]:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return list(set(keywords))

    def extract_keywords(self, resume: Dict[str, any]) -> Dict[str, List[Dict[str, any]]]:
        task = f"""
        Analyze this resume and extract the key skills, qualifications, and experiences:

        Resume: {resume}

        Provide a comprehensive list of keywords, categorized by type (e.g., technical skills, soft skills, etc.). 
        For each keyword, include a relevance score (0-100) based on its prominence and importance in the resume.
        """
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
                keyword, score = content.rsplit('(', 1)
                keyword = keyword.strip()
                score = int(score.rstrip(')').strip())
                parsed_result[current_category].append({"keyword": keyword, "relevance": score})
            elif current_category and line.strip():
                keyword, score = line.rsplit('(', 1)
                keyword = keyword.strip()
                score = int(score.rstrip(')').strip())
                parsed_result[current_category].append({"keyword": keyword, "relevance": score})

        return parsed_result

    def refine_keywords(self, extracted_keywords: Dict[str, List[Dict[str, any]]], job_description: str) -> Dict[
        str, List[Dict[str, any]]]:
        task = f"""
        Refine these extracted resume keywords based on their relevance to the job description:

        Extracted Keywords: {extracted_keywords}

        Job Description: {job_description}

        For each category of keywords:
        1. Adjust the relevance scores based on their importance to the job description.
        2. Remove any keywords that are not relevant to the job.
        3. Suggest additional relevant keywords that might be implied but not explicitly stated in the resume.

        Provide the refined list of keywords with updated relevance scores and any new suggested keywords.
        """
        result = self.execute_task(task)
        return self.parse_result(result.response)
