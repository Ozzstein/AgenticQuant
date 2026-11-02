# Blocker 1: income source acceptance

Status: **open — source sufficiency dependency**, reviewed 2026-11-02.
Read-only IBKR capture intake exists; no positive income parser is registered.
The existing canonical ledger and operating income gate remain authoritative.

## Completion criteria

- [x] Canonical terms, entitlement, receivable, posted-payment and explicitly
  scoped local-paper distribution accounting exist.
- [x] Independent Alpaca wire parsing retains unsupported semantics and unknown
  coverage; source-shaped assertions cannot authorize a cycle.
- [x] Separate event terms, actual settlement, coverage and simulation policy
  requirements, including the mixed equity/crypto scope.
- [ ] Obtain original source delivery plus its documented scope and configuration.
- [ ] Establish a supported security/event/revision identity and pre-ex source
  availability for ordinary dividend terms and ownership convention.
- [ ] Establish actual account posting or actual paid-per-unit occurrence, exact
  deductions/applicability and correction/cancellation linkage.
- [ ] Establish complete required intervals, explicit empty-window semantics,
  continuity, original account/security scope and coverage cutoff.
- [ ] Implement a closed parser from the documented original wire; independently
  reparse it through `verify_income_provenance` without normalized proof flags.
- [ ] Pass source-tamper, chronology, account, revision and completeness tests,
  then the retained fresh operating-cycle acceptance suite.

Completion requires all applicable items. A payment source alone cannot supply
pre-ex announcement history, and an equity feed cannot certify crypto events.

## Reviewed source routes

| Route | Documented component | Remaining acceptance evidence |
| --- | --- | --- |
| Existing Alpaca endpoints | Action/activity observations | Effective-window completeness, linked gross/deductions and entitlement authority remain insufficient; retain unknown coverage |
| IBKR Activity reporting | Account cash movements, accrual amounts and reconciliation ledger | Original configured delivery, reliable event/component joins, corrections, coverage cutoff, and separate pre-ex terms |
| DTCC CA 20022 | Announcement data with event identifiers | Actual licensed feed/profile, scope/continuity and timely revisions; announcement service is not a holder's deduction statement |
| ICE Corporate Actions | Event dates/rates, lifecycle updates, multiple delivery formats | Actual feed specification, entitlement, coverage/replay contract and original messages; product description alone does not certify data |

IBKR's [cash transaction reference](https://www.ibkrguides.com/reportingreference/reportguide/cash%20transactionsfq.htm)
documents account/security identification, time, amount and transaction type.
Its [accrual reference](https://www.ibkrguides.com/reportingreference/reportguide/change%20in%20dividend%20accrualsfq.htm)
documents quantity, gross, net, tax and fee fields, but explicitly allows a
reversal to represent correction, cancellation or payout. Consequently a
reversal cannot independently become a posted payment.

The [Statement of Funds reference](https://www.ibkrguides.com/reportingreference/reportguide/statement%20of%20fundsfq.htm)
describes a period cash ledger with balances and transaction identifiers. This
can support reconciliation. It does not document a universal join from each
dividend to all its deductions or an intraday finality watermark. Do not infer
such a join from matching descriptions, amounts or nearby dates.
[Flex reports are configurable at field level](https://ibkrguides.com/clientportal/performanceandstatements/flex.htm).
Thus an empty section is not sufficient completeness evidence without the exact
included sections, filters, account and period configuration. These are reasons
to inspect an original complete export, not grounds to register IBKR as sufficient.

[DTCC CA 20022](https://www.dtcc.com/products-and-services/data-services/corporate-actions-reference-data/ca-20022-service)
and [ICE Corporate Actions](https://www.ice.com/fixed-income-data-services/data-and-analytics/pricing-and-analytics/reference-data/corporate-actions)
are candidates for the terms-feed component. Their product pages do not provide
the complete source-specific delivery and completeness contracts required here.
The earlier pinned DTCC SR2025 component audit still applies: no general gross,
tax, participant-to-retail allocation or full-window guarantee is inferred.

## Minimum original evidence package

The first intake should cover one ordinary cash dividend in one USD equity and
one exact account, including the preceding announcement and subsequent posting.
It needs:

1. Unmodified provider export/message bodies, original delivery metadata and
   measured acquisition clocks. Availability remains actual; downloading today
   does not establish that a historical capture existed before ex-date.
2. The provider's schema/version, exact report/query configuration, account and
   security mappings, delivery period/timezone and scope/filter settings.
3. Stable business event/revision and transaction/component identifiers, with
   documented correction/reversal behavior. Parser design must follow those
   identifiers rather than invent a description/date matching algorithm.
4. Actual gross/net/fee/tax and quantity evidence, including explicit zero and
   applicability semantics; cash/balance reconciliation and any rounding rules.
5. A coverage/replay contract and delivery evidence supporting the required
   window, including no-event periods and late amendments. Date-only statement
   bounds do not automatically establish readiness for an intraday cycle.

For `posted-account`, the account must be the actual credited account. A broker
payment cannot be assigned to local simulated holdings. For local paper, require
an actual per-unit distribution notice and separately sourced
`PaperDividendPolicy` deduction applicability. Dividing another account's net
payment by its shares does not establish a transferable tax convention.

Mixed crypto cycles additionally require the supported custodial/protocol scope
for their exact instruments. No-staking intent, an empty equity report or an
operator claim that no events occurred cannot establish complete protocol scope.

## Decision and next step

No positive income source has been certified. Original paper-account IBKR bytes
have been acquired as described below. The next external dependency is the
query configuration and sufficient provider delivery specification; no new account,
subscription or credential request is implied. Public documentation review can
shortlist components but cannot substitute for those bytes or access rights.

Once available, implement the narrowest supported profile in the existing source
and operating verifier paths, retain unsupported cases as explicit gaps, and run
the existing canonical economics/coverage/operating tests. Do not add a parallel
income ledger, accept an invented vendor protocol or relax the current gate to
make acceptance fixtures pass. No runtime behavior changed in this review.

## IBKR paper acquisition milestone — 2026-11-02

- [x] Operator retrieved one paper-account Flex XML statement through the
  private 1Password-backed API launcher, with separate paper token/query items.
- [x] Local inspection verified every saved raw-response SHA-256 against its
  receipt and private file modes; one generation and one retrieval completed.
- [x] Inspected section/field structure without publishing account identifiers,
  balances, amounts, transaction descriptions, credentials or raw report bytes.
- [ ] Obtain the live-account report after initial funding/report availability,
  plus exact query configuration and documented delivery scope.
- [ ] Obtain actual income examples and independently prove required identities,
  deductions, revisions, clocks and coverage before positive parser admission.

The paper report contains CashReport, StmtFunds, Trades, CorporateActions,
CashTransactions, ChangeInDividendAccruals, OpenDividendAccruals and SecuritiesInfo
sections. No OpenPositions section is present. Corporate-action and both dividend
accrual sections have no rows; cash rows contain no dividend-classified entry or
nonempty action ID. Thus actual payment/action association and gross/net/deduction
semantics cannot be tested from this sample. Empty sections do not certify a
complete event window. Paper acquisition is working; the live/source sufficiency
blocker remains open. Original XML/receipts stay outside the repository; no
positive source parser, income policy, execution gate or completion criterion was
relaxed. This documentation-only update required no runtime change.

## Bounded IBKR intake implementation — 2026-11-02

- [x] `src.data.ibkr_flex.parse_flex_report(raw, expected_account_id=None)` reads
  original bounded UTF-8 Flex XML into immutable private observations.
- [x] `read_flex_capture(directory, as_of=...)` independently checks local
  acknowledgement/poll/final-report receipts, raw hashes, query identity and
  acquisition chronology, including an original acknowledgement on resumed GETs.
- [x] Decimal strings remain exact; absent fields, blank fields and explicit zero
  remain distinct. Original row strings, IDs, native dates/times and source leaf
  paths are retained; no inferred UTC posting/entitlement clock is created.
- [x] Accrual Po/Re, open accruals, cash and Statement of Funds observations remain
  distinct; no amount/date/description join, gross-minus-net tax inference,
  deduplication, cash creation or canonical entitlement/payment is performed.
- [x] Native SUMMARY rows are retained as `summary_not_posting`, including the
  actual sample's dash account summary. They do not count as separate payments.
- [x] Safe CLI summary exposes fixed section names, counts and gaps without account
  identifiers, descriptions, amounts, provider IDs or native report dates.
- [ ] Register a positive operating source only after the remaining source scope,
  convention, linked deductions, lifecycle and cutoff obligations are supported.

The reader is diagnostic intake, not `verify_income_provenance` admission. Its
`capture_consistency=verified` means local bytes/receipts agree, not that IBKR
cryptographically authenticated them or that the selected account is live.
`source_authentication=not_independently_verified`, `income_admission=blocked` and
zero canonical income records are explicit even for populated synthetic examples.
No new EvidenceStore/ledger authority or source-success flag was introduced.
The original query configuration is not included in the Flex report. Missing
values on native summary/opening-balance lines do not prove a query misconfiguration.
All original records are retained rather than matched or discarded.

Field meanings were checked against the official [cash transaction reference](https://www.ibkrguides.com/reportingreference/reportguide/cash%20transactionsfq.htm),
[change in accruals](https://www.ibkrguides.com/reportingreference/reportguide/change%20in%20dividend%20accrualsfq.htm),
[open accruals](https://www.ibkrguides.com/reportingreference/reportguide/open%20dividend%20accrualsfq.htm)
and [Statement of Funds](https://www.ibkrguides.com/reportingreference/reportguide/statement%20of%20fundsfq.htm).
In particular, reversal code Re is not unique evidence of payout. Literal cash
row labels are observation categories only; unreviewed row types/sections remain
unsupported and never establish economics or complete coverage.

Validation: 104 tests passed in 1.27s, sockets disabled, across the three IBKR test
files, `test_operations_income_proof.py` and `test_action_coverage_accounts.py`.
41 new intake cases exercise real HTTPX fixture capture -> receipt reader,
original-reference continuation, sub-float Decimal precision, summary exclusion,
absent/blank/zero, reversals, corrections and currencies, source corruption,
path aliases/hardlinks, unsafe XML, account scope, future acquisition and redaction.
Test XML is labelled synthetic protocol evidence, not actual payment acceptance.
Ruff check/format and diff check passed. Actual private paper capture was also
read through the implementation; all original capture file hashes stayed unchanged.
No broker request or ledger write was made by this intake implementation.

Private local inspection command (no credentials required):

```bash
.venv/bin/python -m scripts.inspect_ibkr_flex_report --capture /ABSOLUTE/PRIVATE/CAPTURE_FOLDER
```

Next query-configuration review, using the existing saved query rather than
buying assets or generating income:

1. Confirm exact account, reporting period, currency/detail settings and that no
   security, transaction-type or dividend filters unintentionally narrow scope.
2. In Cash Transactions retain account/currency/amount/type/date-time, instrument
   identity and native transaction/action identifiers where IBKR exposes them.
3. In both dividend-accrual sections retain account, currency, conid/security ID,
   ex/pay dates, quantity, gross rate/amount, net amount, tax, fee and code. Change
   in accruals additionally needs its change/report dates where available.
4. Retain Statement of Funds IDs, activity codes, dates and cash balance fields;
   include Corporate Actions, Securities Info and Open Positions for subsequent
   identity/holdings review. Their presence is not itself coverage certification.
5. Keep the original configuration privately with its actual acquisition time.
   An operator confirmation helps review but cannot substitute for a documented
   provider guarantee about completeness, late amendments or tax applicability.

Development can proceed with documentation and labelled tests. A newly opened
account need not have historical dividends; no purchase, deposit or subscription
is required by this implementation. Actual income examples remain a later
acceptance requirement, separate from the now-working paper acquisition/intake.
