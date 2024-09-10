from pydantic import BaseModel
from typing import List, Optional

class JobDescription(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    job_type: Optional[str] = None
    description: str
    requirements: List[str]
    responsibilities: List[str]
    preferred_qualifications: Optional[List[str]] = None
    benefits: Optional[List[str]] = None