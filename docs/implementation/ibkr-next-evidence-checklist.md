# IBKR Flex: next evidence checklist

This is an evidence collection checklist, not an income approval. The Flex intake is diagnostic: it always reports `income_admission: blocked`, creates zero canonical income records, and does not prove live entitlement.

## What to collect

1. Preserve a private, unedited capture directory from the supported acquisition flow, including `capture.json`, original response XML and receipt files. Do not paste report contents into chat or commit them.
2. Separately retain dated evidence of the exact Flex query configuration in effect for that capture: query identity/version, requested date range and timezone, account selections, filters, output/detail/currency settings, and the actual selected fields for each relevant section. The repository does not bind an actual broker configuration export to a capture; the existing query-scope review's declaration is operator-described, not provider-certified. If you cannot establish this configuration-to-capture link, label it unknown and ask the IBKR account operator to preserve the query configuration and capture together. Do not buy a subscription based on this checklist; required product access is unknown from repository evidence.
3. Keep the original exported rows. A field selected in the query is not evidence that the field was returned. In the returned XML, distinguish an absent attribute (not present), blank attribute (present as empty), and a value. A missing field on a populated row is not equivalent to an empty section; an empty/absent section is not proof of no activity or complete coverage.
4. Establish account coverage explicitly: identify every intended account and confirm the capture contains statements for each, with matching account identity and dates. The local reader checks internal statement/row account consistency and local receipt hashes, not account ownership, authorized account coverage, or provider authenticity. Missing account-to-capture mapping remains unknown until the operator supplies independent account/configuration evidence.
5. For each distribution, collect a traceable chain across event/security and applicable terms (including ex/record/pay dates and revision), account and eligible quantity/position, gross amount, each tax/fee/deduction and currency, and posted cash payment with transaction identity/date. Link accrual postings/reversals to the later payment or correction; an accrual is not a payment. Establish whether each deduction applies and its amount, including affirmative zero where applicable. Flex row co-occurrence or a shared security/date is not by itself a validated join. If the source does not provide a verifiable link or all-deduction basis, state that gap and ask the operator to obtain authoritative linked evidence; do not infer it.
6. Preserve amendments and the complete relevant window, including late changes, and document the cutoff. The reader's acquisition timestamp only says when local bytes became available; it does not prove the query window was complete or the event terms were known before ex-date.

## Safe local summary

From the repository root, the existing command accepts only these flags (`--help` verified):

```sh
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m src.data inspect-ibkr-flex --capture /absolute/private/capture-directory
```

Optional deterministic cutoff: `--as-of` followed by an aware ISO 8601 timestamp. The command requires an explicit private capture directory; it does not fetch data or inspect a loose XML file. Its summary reports returned section status and row counts, missing selected values among rows, code-owned record categories, receipt-chain consistency, and fixed blockers. It suppresses account IDs, amounts, native row dates, descriptions and arbitrary text. It does not compare requested fields to returned fields: use the retained configuration evidence and original rows for that comparison. Do not print or share the raw report to make the comparison.

## What repository evidence establishes

- `src/data/ibkr_flex.py` preserves attribute-level `absent` / `blank` / `value`, counts supported sections and rows, and marks required intake values missing on rows. `safe_summary()` reports missing values across returned rows, not query field selections.
- Capture reader checks local receipt/report hashes, acquisition ordering/cutoff, statement date intervals, and internal row/account consistency. These checks are local consistency only, not broker signatures or account coverage certification.
- Income gates in `src/operations/income.py` and `src/research/income.py` require eligible source artifacts and reject unsupported positive authority; Flex is not an admitted income source. Existing synthetic tests can continue checking parser edge cases, absent/blank distinctions, safe summaries, receipt integrity, and fail-closed income gates. Without real income activity, tests cannot establish real payment/deduction linkage, account coverage, provider query completeness, pre-ex terms, live entitlement, or positive income acceptance. Do not manufacture positive fixtures to fill those gaps.
- Unknown dependency: repository evidence does not determine which IBKR plan, entitlement, or subscription is required for the needed report fields. User action: ask the account operator/IBKR to confirm the currently available report fields and access needed for the evidence chain before purchasing anything.
