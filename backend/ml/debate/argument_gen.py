"""LLM-driven argument generation for the Argus Vision visual debate.

This module builds the chat prompts that turn a classifier agent into a
dermatology debater and calls the Groq chat-completions API to realise those
prompts as natural-language arguments. Two rounds are supported:

* **Round 1** — each agent opens by defending its own prediction, citing the
  dermoscopic evidence found in the contested attention region.
* **Round 2** — each agent reads its opponent's opening argument and either
  rebuts and holds, or concedes and revises, emitting a bounded confidence
  delta that the consensus head can fold back into the probabilities.

Every public function is defensive: when no Groq client is supplied, or when the
remote call fails for any reason, a deterministic, clinically grounded fallback
argument is synthesised from :data:`ISIC_CLASS_DESCRIPTIONS` and the region
statistics so the pipeline still completes fully offline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Two-sentence clinical descriptions for each ISIC-8 class, written around the
# dermoscopic criteria a dermatologist would cite when defending a diagnosis.
ISIC_CLASS_DESCRIPTIONS: dict[str, str] = {
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

# Full human-readable names used to make prompts read naturally.
CLASS_FULL_NAMES: dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}

# Regex matching the trailing "CONFIDENCE_DELTA: <float>" instruction line.
_DELTA_PATTERN = re.compile(
    r"CONFIDENCE_DELTA\s*:\s*([+-]?\d*\.?\d+)",
    re.IGNORECASE,
)

# Bounds clamped on any parsed round-2 confidence adjustment.
_DELTA_MIN: float = -0.3
_DELTA_MAX: float = 0.3

# Shared persona reminder reused across both rounds.
_SYSTEM_PERSONA: str = (
    "You are a board-certified dermatology AI participating in a structured "
    "adversarial debate about a single dermoscopic skin-lesion image. You "
    "reason with the ABCDE rule (Asymmetry, Border irregularity, Colour "
    "variegation, Diameter, Evolving) and with established dermoscopic criteria "
    "(pigment network, vessels, dots/globules, structureless zones, "
    "blue-white veil, milia-like cysts, lacunae, keratin). Argue concisely and "
    "persuasively, citing specific visual evidence localised to the contested "
    "region rather than restating textbook definitions. Never invent findings "
    "that contradict the supplied statistics, and never break character."
)


def _full_name(class_code: str) -> str:
    """Return the human-readable name for an ISIC class code.

    Args:
        class_code: One of the canonical ISIC-8 class codes.

    Returns:
        The full diagnosis name, or the raw code if it is unrecognised.
    """
    return CLASS_FULL_NAMES.get(class_code, class_code)


def _describe_class(class_code: str) -> str:
    """Return the clinical description for an ISIC class code.

    Args:
        class_code: One of the canonical ISIC-8 class codes.

    Returns:
        The two-sentence dermoscopic description, or a neutral placeholder if
        the code is unknown.
    """
    return ISIC_CLASS_DESCRIPTIONS.get(
        class_code,
        "No dermoscopic description is available for this lesion class.",
    )


def _stat(region_stats: dict[str, Any], key: str) -> float:
    """Safely extract a numeric statistic from a region-stats mapping.

    Args:
        region_stats: A mapping of attention statistics (e.g. ``mean``,
            ``std``, ``max``).
        key: The statistic to extract.

    Returns:
        The statistic as a ``float``, or ``0.0`` when missing or non-numeric.
    """
    value = region_stats.get(key, 0.0) if region_stats else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_bbox(bbox: Any) -> str:
    """Render a bounding box as ``(x1, y1)-(x2, y2)`` text.

    Accepts either a :class:`~core.models.BoundingBox` (attribute access) or a
    plain mapping with ``x1``/``y1``/``x2``/``y2`` keys.

    Args:
        bbox: The contested-region bounding box.

    Returns:
        A compact human-readable description of the box.
    """
    if bbox is None:
        return "the full lesion (no localised contested region)"

    def _coord(name: str) -> int:
        if isinstance(bbox, dict):
            raw = bbox.get(name, 0)
        else:
            raw = getattr(bbox, name, 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    return (
        f"top-left ({_coord('x1')}, {_coord('y1')}) to "
        f"bottom-right ({_coord('x2')}, {_coord('y2')})"
    )


def _region_summary(region_stats: dict[str, Any], bbox: Any = None) -> str:
    """Build a one-line summary of the contested attention region."""
    return (
        f"In the contested region your attention map has mean {region_stats.get('mean', 0.0):.3f}, "
        f"std {region_stats.get('std', 0.0):.3f}, peak {region_stats.get('max', 0.0):.3f}."
    )


def build_prompt(
    agent_id: str,
    pred_class: str,
    confidence: float,
    region_stats: dict[str, Any],
    bbox: Any,
    opponent_pred: str,
    opponent_conf: float,
    round: int,
) -> list[dict[str, str]]:
    """Build the round-1 chat messages for a single debating agent.

    Args:
        agent_id: This agent's identifier (``"A"`` or ``"B"``).
        pred_class: The ISIC class code this agent predicts.
        confidence: This agent's confidence in ``pred_class`` (0-1).
        region_stats: Attention statistics for this agent inside the contested
            region (expects ``mean``/``std``/``max`` keys).
        bbox: The contested-region bounding box.
        opponent_pred: The opposing agent's predicted ISIC class code.
        opponent_conf: The opposing agent's confidence (0-1).
        round: The debate round number (``1`` for the opening statement).

    Returns:
        A two-element list of chat messages (``system`` then ``user``) ready to
        pass to :func:`generate_argument`.
    """
    user_message = (
        f"You are Agent {agent_id}. You classify this dermoscopic lesion as "
        f"{_full_name(pred_class)} ({pred_class}) with {confidence * 100:.1f}% "
        f"confidence.\n\n"
        f"Clinical profile of {pred_class}: {_describe_class(pred_class)}\n\n"
        f"{_region_summary(region_stats, bbox)}\n\n"
        f"The opposing agent classifies the same lesion as "
        f"{_full_name(opponent_pred)} ({opponent_pred}) with "
        f"{opponent_conf * 100:.1f}% confidence.\n\n"
        f"This is round {round} of the debate. Argue in ONE focused paragraph "
        f"why your diagnosis of {pred_class} is correct. Ground every claim in "
        f"the specific dermoscopic evidence visible inside the contested region "
        f"described above, and explain why those features favour {pred_class} "
        f"over {opponent_pred}. Do not include any preamble, headings, or list "
        f"formatting."
    )
    return [
        {"role": "system", "content": _SYSTEM_PERSONA},
        {"role": "user", "content": user_message},
    ]


def build_counter_prompt(
    agent_id: str,
    pred_class: str,
    original_arg: str,
    opponent_arg: str,
    confidence: float,
    opponent_pred: str,
    opponent_conf: float,
    region_stats: Optional[dict[str, Any]] = None,
    bbox: Any = None,
) -> list[dict[str, str]]:
    """Build the round-2 (rebuttal) chat messages for a debating agent.

    The agent is shown both its own opening argument and the opponent's, then
    asked to rebut and either hold or revise its position, ending with a machine
    -parseable confidence-delta line.

    Args:
        agent_id: This agent's identifier (``"A"`` or ``"B"``).
        pred_class: The ISIC class code this agent originally predicted.
        original_arg: This agent's round-1 argument text.
        opponent_arg: The opponent's round-1 argument text.
        confidence: This agent's current confidence in ``pred_class`` (0-1).
        opponent_pred: The opponent's predicted ISIC class code.
        opponent_conf: The opponent's confidence (0-1).
        region_stats: Optional attention statistics for the contested region.
        bbox: Optional contested-region bounding box.

    Returns:
        A two-element list of chat messages (``system`` then ``user``) ready to
        pass to :func:`generate_argument`.
    """
    region_line = ""
    if region_stats is not None:
        region_line = f"{_region_summary(region_stats, bbox)}\n\n"

    user_message = (
        f"You are Agent {agent_id}, defending a diagnosis of "
        f"{_full_name(pred_class)} ({pred_class}) at {confidence * 100:.1f}% "
        f"confidence.\n\n"
        f"Clinical profile of {pred_class}: {_describe_class(pred_class)}\n\n"
        f"{region_line}"
        f"Your opening argument was:\n\"\"\"\n{original_arg.strip()}\n\"\"\"\n\n"
        f"The opposing agent (predicting {_full_name(opponent_pred)} / "
        f"{opponent_pred} at {opponent_conf * 100:.1f}%) argued:\n"
        f"\"\"\"\n{opponent_arg.strip()}\n\"\"\"\n\n"
        f"This is round 2. In ONE focused paragraph, directly rebut the "
        f"opponent's strongest point using dermoscopic evidence from the "
        f"contested region, then decide whether to HOLD or REVISE your "
        f"confidence in {pred_class}. After the paragraph, on a final separate "
        f"line, output exactly 'CONFIDENCE_DELTA: <number>' where <number> is a "
        f"float in [-0.3, 0.3]: positive if the exchange strengthened your "
        f"diagnosis, negative if the opponent's evidence weakened it, and 0.0 if "
        f"unchanged."
    )
    return [
        {"role": "system", "content": _SYSTEM_PERSONA},
        {"role": "user", "content": user_message},
    ]


def _fallback_argument(
    pred_class: str,
    confidence: float,
    region_stats: Optional[dict[str, Any]],
    bbox: Any = None,
    opponent_pred: Optional[str] = None,
) -> str:
    """Build a deterministic argument when the LLM is unavailable."""
    stats = region_stats or {}
    contrast = ""
    if opponent_pred and opponent_pred != pred_class:
        contrast = (
            f"These features are inconsistent with {_full_name(opponent_pred)} "
            f"({opponent_pred})."
        )
    return (
        f"This lesion is most consistent with {_full_name(pred_class)} ({pred_class}) at "
        f"{confidence * 100:.1f}% confidence. {_describe_class(pred_class)} "
        f"{_region_summary(stats)} {contrast}"
    )


def generate_argument(
    messages: list[dict[str, str]],
    groq_client: Optional[Any],
    model: str,
) -> str:
    """Generate an argument paragraph from chat messages via Groq.

    This issues a **synchronous** Groq chat-completions request; the calling
    pipeline is responsible for off-loading it to a worker thread. Any failure
    (missing client, network error, malformed response) degrades gracefully to a
    deterministic fallback derived from the supplied user message.

    Args:
        messages: Chat messages produced by :func:`build_prompt`.
        groq_client: An initialised Groq SDK client, or ``None`` to force the
            offline fallback.
        model: The Groq model identifier to query.

    Returns:
        The stripped argument text from the model, or a deterministic fallback
        paragraph when generation is not possible.
    """
    if groq_client is None:
        logger.warning("Groq client unavailable; using fallback argument.")
        return _fallback_from_messages(messages)

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            logger.warning("Groq returned empty content; using fallback.")
            return _fallback_from_messages(messages)
        return content.strip()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any error.
        logger.warning("Groq argument generation failed (%s); using fallback.", exc)
        return _fallback_from_messages(messages)


def _fallback_from_messages(messages: list[dict[str, str]]) -> str:
    """Derive a minimal fallback argument directly from a message list.

    Used when the structured statistics are not in scope (the messages are the
    only available context). It echoes the substantive user content into a
    single, character-preserving paragraph.

    Args:
        messages: The chat messages that were going to be sent to the LLM.

    Returns:
        A deterministic fallback paragraph.
    """
    user_text = ""
    for message in messages:
        if message.get("role") == "user":
            user_text = message.get("content", "")
            break
    if not user_text:
        return (
            "Argument generation is temporarily unavailable; the agent maintains "
            "its original prediction based on the observed dermoscopic features."
        )
    # Collapse whitespace and keep the clinically meaningful portion.
    condensed = " ".join(user_text.split())
    return (
        "Argument generation is temporarily unavailable, so this is an offline "
        "summary of the agent's position. " + condensed
    )


def generate_counter_argument(
    original_arg: str,
    opponent_arg: str,
    pred_class: str,
    groq_client: Optional[Any],
    model: str,
    agent_id: str = "",
    confidence: float = 0.0,
    opponent_pred: str = "",
    opponent_conf: float = 0.0,
    region_stats: Optional[dict[str, Any]] = None,
    bbox: Any = None,
) -> tuple[str, float]:
    """Generate a round-2 rebuttal and a bounded confidence delta."""
    if groq_client is None:
        fb = _fallback_argument(
            pred_class=pred_class,
            confidence=confidence,
            region_stats=region_stats,
            bbox=bbox,
            opponent_pred=opponent_pred,
        )
        return fb, 0.0

    messages = build_counter_prompt(
        agent_id=agent_id,
        pred_class=pred_class,
        original_arg=original_arg,
        opponent_arg=opponent_arg,
        confidence=confidence,
        opponent_pred=opponent_pred,
        opponent_conf=opponent_conf,
        region_stats=region_stats,
        bbox=bbox,
    )

    raw = generate_argument(messages, groq_client, model)
    return _split_delta(raw)


def _split_delta(text: str) -> tuple[str, float]:
    """Separate the argument body from its trailing confidence-delta line.

    Args:
        text: The raw LLM (or fallback) reply, optionally ending with a
            ``CONFIDENCE_DELTA: <float>`` line.

    Returns:
        A ``(argument, delta)`` tuple where ``argument`` has the delta line
        removed and ``delta`` is clamped to ``[-0.3, 0.3]`` (``0.0`` if absent).
    """
    match = _DELTA_PATTERN.search(text)
    delta = 0.0
    if match is not None:
        try:
            delta = float(match.group(1))
        except (TypeError, ValueError):
            delta = 0.0
    delta = max(_DELTA_MIN, min(_DELTA_MAX, delta))

    # Remove every occurrence of the delta line from the visible argument.
    argument = _DELTA_PATTERN.sub("", text).strip()
    if not argument:
        argument = text.strip()
    return argument, delta
