from datetime import UTC, date, datetime


def utc_now():
    return datetime.now(UTC)


def data_age(as_of, now):
    return (now.astimezone(UTC).date() - date.fromisoformat(as_of)).days


def report_freshness(report, settings, now):
    """Fail closed for legacy, replay, future-dated or expired reports."""
    try:
        generated = datetime.fromisoformat(report["generated_at"])
        if generated.tzinfo is None:
            raise ValueError("Report timestamp must have a timezone")
        age = (now - generated).total_seconds()
        days = data_age(report["as_of"], now)
        fresh = (
            report.get("mode") == "live"
            and 0 <= age <= settings.max_report_age_seconds
            and 0 <= days <= settings.max_data_age_days
        )
        return {"fresh": fresh, "report_age_seconds": age, "data_age_days": days}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"fresh": False, "report_age_seconds": None, "data_age_days": None}
