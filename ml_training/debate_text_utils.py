# Shared debate text utilities to prevent train-test phrasing mismatches.

CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}

ISIC_CLASS_DESCRIPTIONS = {
    "MEL": "Melanoma typically shows an atypical, broadened pigment network with irregular streaks, "
           "asymmetry of structure and colour, and frequent regression areas. A blue-white veil and "
           "chaotic vessels support malignancy.",
    "NV": "A melanocytic nevus is characterised by a symmetric, regularly spaced reticular or globular "
          "pattern with uniform colouration and a smooth transition to surrounding skin.",
    "BCC": "Basal cell carcinoma is defined by arborising (tree-like) vessels and blue-grey ovoid nests "
           "on a pigment-network-free background, with leaf-like areas and spoke-wheel structures.",
    "AK": "Actinic keratosis shows a 'strawberry' pattern: a red pseudo-network of dilated vessels around "
          "keratin-plugged follicular openings on a scaly, erythematous background.",
    "BKL": "Benign keratosis displays a cerebriform 'brain-like' surface with milia-like cysts and "
           "comedo-like openings, sharply demarcated borders and a stuck-on appearance.",
    "DF": "Dermatofibroma presents with a central white scar-like patch surrounded by a delicate peripheral "
          "pigment network and a homogeneous tan-brown ring.",
    "VASC": "Vascular lesions are recognised by sharply demarcated red, purple, or maroon lacunae separated "
            "by pale septa, with no melanocytic pigment network.",
    "SCC": "Squamous cell carcinoma shows central keratin masses, white circles around follicular openings, "
           "surface scale/ulceration, and looped or glomerular vessels at the periphery.",
}

def _region_summary(stats):
    return (f"In the contested region your attention map has mean {stats.get('mean', 0.0):.3f}, "
            f"std {stats.get('std', 0.0):.3f}, peak {stats.get('max', 0.0):.3f}.")

def _fallback_argument(pred, conf, stats, opponent):
    return (f"This lesion is most consistent with {CLASS_FULL_NAMES[pred]} ({pred}) at {conf * 100:.1f}% "
            f"confidence. {ISIC_CLASS_DESCRIPTIONS[pred]} {_region_summary(stats)} These features are "
            f"inconsistent with {CLASS_FULL_NAMES.get(opponent, opponent)} ({opponent}).")
