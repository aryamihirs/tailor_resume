from pydantic import BaseModel, Field

class SectionConstraint(BaseModel):
    max_tokens: int = Field(..., ge=0)

class ExperienceConstraint(BaseModel):
    single_line: SectionConstraint
    double_line: SectionConstraint

class Constraints(BaseModel):
    about_me: SectionConstraint
    skills: SectionConstraint
    experiences: ExperienceConstraint
    projects: ExperienceConstraint