import asyncio
import json
from pathlib import Path

from app.classifier import build_default_classifier
from app.schemas import IncidentCreate


async def main() -> None:
    tickets_path = Path(__file__).resolve().parent / "synthetic_tickets.json"
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    classifier = build_default_classifier()

    total = len(tickets)
    passed = 0

    for idx, item in enumerate(tickets, start=1):
        incident = IncidentCreate(
            store_id=item["store_id"],
            title=item["title"],
            description=item["description"],
        )
        result = await classifier.classify(incident)
        actual_cat = result.suggested_category.value if result.suggested_category else None
        ok = (
            actual_cat == item["expected_category"]
            and result.needs_review == item["expected_needs_review"]
        )
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] #{idx:02d} {item['store_id']} | "
            f"cat={actual_cat} (exp={item['expected_category']}) | "
            f"review={result.needs_review} (exp={item['expected_needs_review']})"
        )

    print(f"\nEvaluation Summary: {passed}/{total} tickets matched ({passed / total * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
