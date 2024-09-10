# agents/keyword_similarity_ranking.py

from .base_react_agent import BaseReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KeywordSimilarityRankingAgent(BaseReActAgent):
    def __init__(self):
        system_prompt = """
        You are an expert in analyzing and ranking keyword similarities between job descriptions and resumes. 
        Your task is to compare keywords from a resume with those from a job description, determining their 
        relevance and similarity. Consider not just exact matches, but also semantic similarities and variations.

        For each keyword comparison, consider:
        1. Exact matches
        2. Semantic similarities
        3. Industry-specific synonyms
        4. Variations in terminology (e.g., "Python programming" vs "Python developer")
        5. The importance of the keyword in the context of the job description

        Provide a ranked list of resume keywords based on their similarity and relevance to the job description keywords.
        Include a similarity score and brief explanation for each keyword match.
        """
        super().__init__("KeywordSimilarityRanking",
                         "Rank resume keywords based on similarity to job description keywords", system_prompt)

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.rank_keywords,
                name="rank_keywords",
                description="Rank resume keywords based on similarity to job description keywords"
            )
        ]

    def rank_keywords(self, resume_keywords: List[str], job_keywords: List[str]) -> List[Dict[str, any]]:
        # First, use TF-IDF and cosine similarity for initial ranking
        tfidf = TfidfVectorizer()
        all_keywords = resume_keywords + job_keywords
        tfidf_matrix = tfidf.fit_transform(all_keywords)

        resume_vectors = tfidf_matrix[:len(resume_keywords)]
        job_vectors = tfidf_matrix[len(resume_keywords):]

        similarities = cosine_similarity(resume_vectors, job_vectors)

        # Create initial ranking based on TF-IDF similarity
        initial_ranking = [
            {"keyword": kw, "similarity": similarities[i].mean()}
            for i, kw in enumerate(resume_keywords)
        ]
        initial_ranking.sort(key=lambda x: x["similarity"], reverse=True)

        # Use AI to refine the ranking and provide explanations
        task = f"""
        Refine this keyword similarity ranking and provide explanations:

        Resume Keywords: {resume_keywords}
        Job Keywords: {job_keywords}
        Initial Ranking: {initial_ranking}

        For each keyword in the initial ranking:
        1. Adjust the similarity score if needed based on semantic similarity and context.
        2. Provide a brief explanation for the similarity or relevance.
        3. Suggest any potential keyword variations or synonyms that might be more relevant.

        Return the refined ranking with explanations and suggestions.
        """
        result = self.execute_task(task)
        return self.parse_result(result.response)

    def parse_result(self, result: str) -> List[Dict[str, any]]:
        lines = result.strip().split('\n')
        parsed_result = []
        current_keyword = None

        for line in lines:
            if line.startswith("Keyword:"):
                if current_keyword:
                    parsed_result.append(current_keyword)
                current_keyword = {"keyword": line.split(":", 1)[1].strip()}
            elif line.startswith("Similarity:"):
                current_keyword["similarity"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Explanation:"):
                current_keyword["explanation"] = line.split(":", 1)[1].strip()
            elif line.startswith("Suggestions:"):
                current_keyword["suggestions"] = [s.strip() for s in line.split(":", 1)[1].split(",")]

        if current_keyword:
            parsed_result.append(current_keyword)

        return parsed_result
