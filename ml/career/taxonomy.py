"""Major families predicted by the ML model and the rules that map free-text
majors from the RIASEC dataset onto them.

Families are deliberately coarser than the 40 career profiles of the app:
6 RIASEC scale scores cannot separate e.g. "cloud" from "software", but they
can separate software from nursing. The app maps each profile to one family
(see FAMILY_OF_PROFILE) and refines inside the family with its own rules.
"""
import re

FAMILIES = {
    "it_software": "Công nghệ thông tin",
    "engineering": "Kỹ thuật",
    "architecture_construction": "Kiến trúc & Xây dựng",
    "business_management": "Kinh doanh & Quản trị",
    "marketing_communication": "Marketing & Truyền thông",
    "finance_economics": "Tài chính, Kế toán & Kinh tế",
    "arts_design": "Nghệ thuật & Thiết kế",
    "humanities_language": "Ngôn ngữ & Nhân văn",
    "psychology_counseling": "Tâm lý học",
    "education": "Giáo dục",
    "social_work": "Xã hội học & Công tác xã hội",
    "law_public": "Luật & Khu vực công",
    "health_medicine": "Y tế & Điều dưỡng",
    "life_sciences": "Khoa học sự sống",
    "physical_sciences_math": "Khoa học tự nhiên & Toán",
    "hospitality_tourism": "Du lịch & Dịch vụ",
    "sports_fitness": "Thể thao & Vận động",
}

# Order matters: first match wins, so specific patterns come before generic ones.
_RULES = [
    ("it_software", r"computer|computing|software|information (technology|systems?)|\bit\b|\bi\.t\b|programming|data science|cyber|informatics|web (design|dev)|network"),
    ("architecture_construction", r"architect|construction|civil eng|urban plan|interior design|quantity survey|building"),
    ("engineering", r"engineer|mechatronic|aerospace|aeronaut|aviation|automotive|electronics?|electrical|robotics"),
    ("finance_economics", r"account|financ|econom|bank|actuar|audit|\bcpa\b|tax"),
    ("marketing_communication", r"marketing|advertis|public relations|\bpr\b|communication|journalism|media|broadcast"),
    ("hospitality_tourism", r"hospitality|touris|hotel|culinary|chef|cook|travel|event management"),
    ("sports_fitness", r"kinesiolog|exercise|sport|physical education|fitness|athlet"),
    ("business_management", r"business|bu[is]{1,3}ness|\bbba\b|management|commerce|\bmba\b|human resource|\bhrm?\b|leadership|entrepreneur|logistic|supply chain|retail|real estate"),
    ("arts_design", r"\bart\b|\barts\b(?! and science)|fine art|design|music|film|theat|drama|animation|photograph|dance|fashion|illustrat|painting|creative writing"),
    ("psychology_counseling", r"ps[yi]ch|pysch|psyc|counsel|therap(y|ist)(?!.*physical)|behavio"),
    ("social_work", r"social work|social services|human services|youth work|community (development|service)|non.?profit|sociolog|anthropolog|archaeolog|social science"),
    ("education", r"educat|teach|pedagog|early childhood|child development|human development|family studies|tesol|tefl"),
    ("law_public", r"\blaw\b|legal|criminal|criminolog|justice|politic|public (admin|policy|service)|international (relations|studies)|government|police|military|paralegal"),
    ("health_medicine", r"nurs|medic|pre-?med|doctor|speech (pathology|therapy)|rehabilitation|pharma|public health|health|dent|physio|physical therap|occupational therap|paramedic|radiolog|midwif|optometr|nutrition|dietet|surgery|\bmd\b|mbbs|veterin"),
    ("life_sciences", r"biolog|biochem|neuro|zoolog|marine|ecolog|environment|agricultur|horticultur|food (science|technology)|botan|genetic|microbio|forensic|animal|forestry|biotech"),
    ("physical_sciences_math", r"chemi|physic|math|statistic|geolog|astronom|geograph|earth science|science"),
    ("humanities_language", r"english|literature|linguist|language|history|philosoph|classics|theolog|relig|french|spanish|german|japanese|chinese|translation|writing|humanities|liberal (arts|studies)"),
]
_COMPILED = [(family, re.compile(pattern)) for family, pattern in _RULES]

_NON_ANSWERS = re.compile(r"^(no|none|n/?a|na|-+|\.|undecided|undeclared|nothing|not applicable|yes|0|x|general( studies)?|unknown|other)$")


def map_major(text) -> str | None:
    """Map a free-text major to a family, or None when it is empty/ambiguous."""
    if not isinstance(text, str):
        return None
    value = re.sub(r"\s+", " ", text.strip().lower())
    if len(value) < 2 or _NON_ANSWERS.match(value):
        return None
    for family, pattern in _COMPILED:
        if pattern.search(value):
            return family
    return None


# Each of the app's 40 career profiles belongs to one ML family (or the sum of a few).
FAMILY_OF_PROFILE = {
    "software_it": "it_software",
    "ai_data": "it_software",
    "cybersecurity_network": "it_software",
    "cloud_devops": "it_software",
    "game_interactive": "it_software",
    "education_technology": ["education", "it_software"],
    "engineering_mechanical": "engineering",
    "engineering_electrical": "engineering",
    "automotive_transport": "engineering",
    "energy_renewable": "engineering",
    "aviation_maritime": "engineering",
    "construction_architecture": "architecture_construction",
    "geospatial_gis": "architecture_construction",
    "business_management": "business_management",
    "logistics_supplychain": "business_management",
    "entrepreneurship_innovation": "business_management",
    "ecommerce_digital": "business_management",
    "hr_people_ops": "business_management",
    "marketing_media": "marketing_communication",
    "content_media": "marketing_communication",
    "journalism_investigation": "marketing_communication",
    "event_creative": "marketing_communication",
    "finance_accounting": "finance_economics",
    "economics_research": "finance_economics",
    "design_multimedia": "arts_design",
    "product_ux_research": "arts_design",
    "language_international": "humanities_language",
    "public_diplomacy": "law_public",
    "law_publicservice": "law_public",
    "psychology_education": ["psychology_counseling", "education"],
    "social_impact": "social_work",
    "healthcare_biology": "health_medicine",
    "food_nutrition": "health_medicine",
    "biotech_pharma": "life_sciences",
    "environment_agriculture": "life_sciences",
    "veterinary_animal": "life_sciences",
    "science_research": "physical_sciences_math",
    "chemistry_materials": "physical_sciences_math",
    "hospitality_tourism": "hospitality_tourism",
    "sports_wellness": "sports_fitness",
}


def families_of(profile: str) -> list[str]:
    value = FAMILY_OF_PROFILE.get(profile, [])
    return [value] if isinstance(value, str) else list(value)


assert {f for p in FAMILY_OF_PROFILE for f in families_of(p)} == set(FAMILIES)
