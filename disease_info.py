"""Reference text for PlantVillage-style class names ("Crop___Condition").

IMPORTANT: these summaries are general, deliberately cautious descriptions written for this
project from standard plant-pathology knowledge. Before submitting the project, check each
entry against a source you cite (e.g. your state agricultural-university extension factsheets,
or PlantVillage disease pages) and replace/extend the text as needed. Recommendations are general
and never name specific pesticides or doses.
"""
from __future__ import annotations

import re

SOURCES = [
    "Hughes & Salathé (2015). An open access repository of images on plant health to enable the "
    "development of mobile disease diagnostics. arXiv:1511.08060 (PlantVillage dataset).",
    "Mohanty, Hughes & Salathé (2016). Using deep learning for image-based plant disease detection. "
    "Frontiers in Plant Science 7:1419.",
    "General plant-pathology references: consult your local agricultural university / extension service "
    "for crop- and region-specific guidance.",
]

GENERAL_TIPS = [
    "Confirm the problem with a local agricultural extension officer or plant pathologist before treating.",
    "Isolate or mark the affected plants and avoid moving tools or hands from sick plants to healthy ones.",
]

# key -> (disease name, cause, description, tips)
_INFO = {
    "healthy": ("Healthy", "—",
        "No visible disease symptoms were recognised on this leaf.",
        ["Keep scouting leaves regularly, especially after rain or humid spells.",
         "Water at the base of plants and keep good airflow between them.",
         "Maintain balanced nutrition and clean up fallen plant debris."]),
    "scab": ("Scab", "Fungus (Venturia inaequalis)",
        "Olive-green to brown, velvety or scabby spots on leaves and fruit; heavy infection can cause early leaf drop.",
        ["Rake up and remove fallen leaves in autumn to reduce spores.", "Prune for open canopies so leaves dry quickly.",
         "Ask your extension service about resistant varieties and spray timing."]),
    "black_rot": ("Black rot", "Fungus (species depends on the crop)",
        "Dark, often circular leaf lesions that can progress to fruit rot and shrivelled, mummified fruit.",
        ["Remove mummified fruit and infected shoots; do not leave them in the field.",
         "Improve airflow through pruning and trellising.", "Ask your extension service about protective programmes."]),
    "cedar": ("Cedar apple rust", "Fungus (Gymnosporangium juniperi-virginianae)",
        "Bright yellow-orange spots on leaves, sometimes with small tube-like structures underneath. The fungus alternates with juniper/red cedar.",
        ["Remove nearby galls on juniper where practical.", "Choose rust-resistant varieties.", "Ask your extension service about timing of protective measures."]),
    "powdery": ("Powdery mildew", "Fungus (several species)",
        "White, powdery coating on leaf surfaces; leaves may curl, yellow or drop.",
        ["Improve air circulation and avoid excess nitrogen.", "Remove badly affected leaves.", "Prefer resistant varieties where available."]),
    "gray_leaf": ("Gray leaf spot", "Fungus (Cercospora zeae-maydis)",
        "Narrow, rectangular tan-to-gray lesions running between leaf veins; favoured by warm, humid weather and maize residue.",
        ["Rotate crops and manage crop residue.", "Plant tolerant hybrids.", "Avoid dense planting that keeps leaves wet."]),
    "common_rust": ("Common rust", "Fungus (Puccinia sorghi)",
        "Small cinnamon-brown pustules on both leaf surfaces that release powdery spores.",
        ["Use resistant hybrids.", "Scout early in the season; severe cases are most important on young crops.", "Ask your extension service whether treatment is justified."]),
    "northern": ("Northern leaf blight", "Fungus (Exserohilum turcicum)",
        "Long, cigar-shaped gray-green to tan lesions on maize leaves, favoured by moderate temperatures and humid, wet weather.",
        ["Rotate crops and bury or remove residue.", "Plant resistant hybrids.", "Scout before and after tasselling."]),
    "esca": ("Esca (black measles)", "Fungal complex of wood-decay organisms",
        "Leaves show tiger-stripe patterns of yellow/red and dead tissue; berries may spot. It is a trunk disease with no simple cure.",
        ["Mark affected vines and prune in dry weather.", "Protect pruning wounds as advised locally.", "Remove and destroy dead wood."]),
    "isariopsis": ("Leaf blight (Isariopsis leaf spot)", "Fungus (Pseudocercospora vitis)",
        "Angular to irregular dark-brown spots on grape leaves, often on older leaves, that can cause early defoliation.",
        ["Improve canopy airflow and sunlight penetration.", "Clear fallen leaves.", "Ask your extension service about control options."]),
    "greening": ("Citrus greening (Huanglongbing)", "Bacterium (Candidatus Liberibacter) spread by psyllids",
        "Blotchy, asymmetrical yellowing of leaves, small lopsided fruit and decline. There is no cure and it is a serious regulated disease in many regions.",
        ["Report suspected cases to your agricultural authority.", "Control the psyllid vector as advised locally.", "Use certified disease-free nursery trees."]),
    "bacterial_spot": ("Bacterial spot", "Bacteria (Xanthomonas species)",
        "Small, dark, water-soaked spots on leaves (sometimes with a yellow halo) that may spread to fruit; worse in warm, wet weather.",
        ["Use certified disease-free seed or plants.", "Avoid overhead irrigation and handling wet plants.", "Rotate crops and remove infected debris."]),
    "early_blight": ("Early blight", "Fungus (Alternaria solani)",
        "Brown spots with concentric 'target' rings, usually starting on older, lower leaves, often surrounded by yellowing.",
        ["Rotate crops and remove infected debris.", "Mulch and water at the base to limit splash.", "Keep plants well fed and not water-stressed."]),
    "late_blight": ("Late blight", "Oomycete (Phytophthora infestans)",
        "Water-soaked gray-green lesions that darken quickly, sometimes with white growth underneath in humid conditions. It can destroy a crop within days in cool, wet weather.",
        ["Act quickly: seek local advice as soon as possible.", "Remove and destroy infected plants; do not compost them.",
         "Avoid overhead watering and improve airflow."]),
    "leaf_mold": ("Leaf mold", "Fungus (Passalora fulva)",
        "Pale yellow patches on upper leaf surfaces with olive-gray mold underneath; common in humid greenhouses.",
        ["Reduce humidity and increase ventilation.", "Space and prune plants for airflow.", "Use resistant varieties where available."]),
    "septoria": ("Septoria leaf spot", "Fungus (Septoria lycopersici)",
        "Many small circular spots with dark borders and gray centres, usually on lower leaves first.",
        ["Remove affected lower leaves.", "Mulch and avoid wetting foliage.", "Rotate crops and clear debris."]),
    "spider": ("Spider mite damage", "Two-spotted spider mite (Tetranychus urticae)",
        "Fine yellow stippling on leaves, sometimes with silky webbing; thrives in hot, dry conditions.",
        ["Check leaf undersides with a hand lens.", "Reduce dust and water stress.", "Ask your extension service about mite management."]),
    "target": ("Target spot", "Fungus (Corynespora cassiicola)",
        "Brown lesions with concentric rings and yellow margins on leaves, sometimes on fruit.",
        ["Improve airflow and avoid long leaf-wetness periods.", "Remove infected debris.", "Rotate crops."]),
    "ylcv": ("Yellow leaf curl virus", "Virus (TYLCV) spread by whiteflies",
        "Upward-curling, yellowing leaves and stunted growth. Viral diseases cannot be cured.",
        ["Control whiteflies and use insect-proof nets where possible.", "Remove and destroy infected plants.", "Choose resistant varieties."]),
    "mosaic": ("Mosaic virus", "Virus (Tomato mosaic virus), spread mechanically",
        "Mottled light and dark green patches, distorted leaves. It spreads on hands, tools and seed and cannot be cured.",
        ["Wash hands and disinfect tools between plants.", "Remove infected plants.", "Use certified seed and resistant varieties."]),
    "leaf_scorch": ("Leaf scorch", "Fungus (Diplocarpon earlianum)",
        "Many small purple spots that merge, giving leaves a scorched look.",
        ["Remove old infected leaves after harvest.", "Give plants adequate spacing.", "Avoid overhead watering."]),
}

# Ordered (first match wins). Patterns match the lower-cased condition part of the class name.
_RULES = [
    ("healthy", "healthy"), ("cedar", "cedar"), ("scab", "scab"), ("black_rot", "black_rot"),
    ("powdery", "powdery"), ("gray_leaf|cercospora", "gray_leaf"), ("common_rust", "common_rust"),
    ("northern", "northern"), ("esca|measles", "esca"), ("isariopsis|leaf_blight", "isariopsis"),
    ("haunglongbing|greening", "greening"), ("bacterial_spot", "bacterial_spot"),
    ("early_blight", "early_blight"), ("late_blight", "late_blight"), ("leaf_mold", "leaf_mold"),
    ("septoria", "septoria"), ("spider|mite", "spider"), ("target", "target"),
    ("yellow_leaf_curl", "ylcv"), ("mosaic", "mosaic"), ("scorch", "leaf_scorch"),
]


def split_class(label: str) -> tuple[str, str]:
    crop, _, cond = label.partition("___")
    return crop, cond


def pretty(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    return text[:1].upper() + text[1:]


def display_name(label: str) -> str:
    crop, cond = split_class(label)
    return f"{pretty(crop)}: {pretty(cond)}" if cond else pretty(label)


def lookup(label: str) -> dict:
    """Return reference info for a class label, or an explicit 'no reference' entry."""
    crop, cond = split_class(label)
    key_text = re.sub(r"[^a-z0-9]+", "_", cond.lower()).strip("_")
    for pattern, key in _RULES:
        if re.search(pattern, key_text):
            name, cause, desc, tips = _INFO[key]
            return {"found": True, "crop": pretty(crop), "condition": name, "cause": cause,
                    "description": desc, "tips": tips + GENERAL_TIPS, "healthy": key == "healthy"}
    return {"found": False, "crop": pretty(crop), "condition": pretty(cond or label), "cause": "—",
            "description": "No reference text is available for this class yet. Add it to disease_info.py.",
            "tips": GENERAL_TIPS, "healthy": False}
