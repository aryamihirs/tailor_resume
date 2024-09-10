# orchestrator/main_orchestrator.py

from typing import Dict, Any
from ..models.resume import Resume
from ..models.job_description import JobDescription
from ..models.constraints import Constraints
from ..agents.constraint_inference import ConstraintInferenceAgent
from ..agents.keyword_extraction import JobDescriptionKeywordExtractionAgent, ResumePointKeywordExtractionAgent
from ..agents.keyword_similarity import KeywordSimilarityRankingAgent
from ..agents.resume_point_tailoring import ResumePointTailoringAgent
from ..agents.ats_score_estimation import ATSScoreEstimationAgent
from ..agents.human_readability import HumanReadabilityAgent
from ..agents.resume_coherence import ResumeCoherenceAgent
from ..agents.industry_context import IndustryContextAgent
from ..agents.format_compliance import FormatComplianceAgent
from ..agents.soft_skills_balancing import SoftSkillsBalancingAgent
from ..agents.customization import CustomizationAgent
from ..agents.application_context import ApplicationContextAgent


class MainOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize_agents()

    def initialize_agents(self):
        self.constraint_inference_agent = ConstraintInferenceAgent()
        self.job_keyword_extraction_agent = JobDescriptionKeywordExtractionAgent()
        self.resume_keyword_extraction_agent = ResumePointKeywordExtractionAgent()
        self.keyword_similarity_agent = KeywordSimilarityRankingAgent()
        self.resume_point_tailoring_agent = ResumePointTailoringAgent()
        self.ats_score_estimation_agent = ATSScoreEstimationAgent()
        self.human_readability_agent = HumanReadabilityAgent()
        self.resume_coherence_agent = ResumeCoherenceAgent()
        self.industry_context_agent = IndustryContextAgent()
        self.format_compliance_agent = FormatComplianceAgent()
        self.soft_skills_balancing_agent = SoftSkillsBalancingAgent()
        self.customization_agent = CustomizationAgent()
        self.application_context_agent = ApplicationContextAgent()

    def tailor_resume(self, resume: Resume, job_description: JobDescription) -> Resume:
        # Step 1: Infer constraints
        constraints = self.constraint_inference_agent.infer_constraints(resume.dict())

        # Step 2: Extract keywords from job description
        job_keywords = self.job_keyword_extraction_agent.extract_keywords(job_description.description)

        # Step 3: Get industry context
        industry_context = self.industry_context_agent.get_industry_context(job_description.description,
                                                                            job_description.company)

        # Step 4: Analyze application context
        application_context = self.application_context_agent.analyze_context("",
                                                                             "")  # Placeholder for cover letter and portfolio

        # Step 5: Tailor each section of the resume
        tailored_resume = self.tailor_resume_sections(resume, job_keywords, constraints, industry_context,
                                                      application_context)

        # Step 6: Balance soft skills
        tailored_resume = self.soft_skills_balancing_agent.balance_soft_skills(tailored_resume.dict())

        # Step 7: Ensure format compliance
        tailored_resume = self.format_compliance_agent.check_format(tailored_resume)

        # Step 8: Check resume coherence
        coherence_result = self.resume_coherence_agent.check_coherence(tailored_resume)

        # Step 9: Estimate ATS score
        ats_score = self.ats_score_estimation_agent.estimate_ats_score(tailored_resume)

        # Step 10: Check human readability
        readability_score = self.human_readability_agent.check_readability(str(tailored_resume))

        # Step 11: Customize based on user preferences (placeholder)
        tailored_resume = self.customization_agent.customize_resume(tailored_resume, {}, job_description.company)

        return Resume(**tailored_resume)

    def tailor_resume_sections(self, resume: Resume, job_keywords: list, constraints: Constraints,
                               industry_context: dict, application_context: dict) -> Resume:
        tailored_resume = resume.dict()

        for section in ['summary', 'experiences', 'projects', 'skills']:
            if isinstance(tailored_resume[section], list):
                tailored_resume[section] = [
                    self.tailor_section_item(item, job_keywords, constraints, industry_context, application_context) for
                    item in tailored_resume[section]]
            else:
                tailored_resume[section] = self.tailor_section_item(tailored_resume[section], job_keywords, constraints,
                                                                    industry_context, application_context)

        return Resume(**tailored_resume)

    def tailor_section_item(self, item: Any, job_keywords: list, constraints: Constraints, industry_context: dict,
                            application_context: dict) -> Any:
        if isinstance(item, dict):
            item_keywords = self.resume_keyword_extraction_agent.extract_keywords(str(item))
            keyword_mapping = self.keyword_similarity_agent.rank_keywords(item_keywords, job_keywords)
            tailored_item = self.resume_point_tailoring_agent.tailor_point(str(item), keyword_mapping,
                                                                           constraints.dict())
            return tailored_item
        elif isinstance(item, str):
            item_keywords = self.resume_keyword_extraction_agent.extract_keywords(item)
            keyword_mapping = self.keyword_similarity_agent.rank_keywords(item_keywords, job_keywords)
            tailored_item = self.resume_point_tailoring_agent.tailor_point(item, keyword_mapping, constraints.dict())
            return tailored_item
        else:
            return item

    def run(self, resume: Resume, job_description: JobDescription) -> Resume:
        tailored_resume = self.tailor_resume(resume, job_description)
        return tailored_resume
