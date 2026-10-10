def quality_gate(
    candidate, baseline, incumbent, improvement, segment_regression, latency_seconds, max_seconds
):
    reasons = []
    score = candidate["overall"]["wape"]
    if score is None:
        reasons.append("candidate WAPE undefined: zero actual demand")
    for label, reference in [("baseline", baseline), ("incumbent", incumbent)]:
        if reference is None:
            continue
        ref_score = reference["overall"]["wape"]
        if score is None or ref_score is None or not score < ref_score * (1 - improvement):
            reasons.append(f"requires {improvement:.0%} WAPE improvement over {label}")
        for segment, value in reference.items():
            if segment == "overall":
                continue
            new = candidate.get(segment, {}).get("mae")
            if new is None or new > value["mae"] * (1 + segment_regression) + 1e-9:
                reasons.append(f"{label} segment regression: {segment}")
    if latency_seconds > max_seconds:
        reasons.append("batch prediction latency exceeded threshold")
    return {"passed": not reasons, "reasons": reasons}
