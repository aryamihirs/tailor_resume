from pydantic import BaseModel, Field
from typing import List, Optional

class Skill(BaseModel):
    name: str
    level: Optional[str] = None

class Experience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: Optional[str] = "Present"
    description: List[str]

class Project(BaseModel):
    name: str
    description: str
    technologies: List[str]
    url: Optional[str] = None

class Education(BaseModel):
    degree: str
    institution: str
    graduation_date: str

class Resume(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: str = Field(..., max_length=500)
    skills: List[Skill]
    experiences: List[Experience]
    projects: List[Project]
    education: List[Education]
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None