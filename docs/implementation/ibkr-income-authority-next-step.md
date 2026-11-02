# IBKR income authority: bounded next step

Reviewed 2026-11-02 against committed base
`4695f786127f5a5c0fd339ddad74058396e457df`.

**Decision:** keep Flex as diagnostic account observations. The configured query
and empty new-account statement are useful evidence of intake readiness, but do
not yet justify a positive income parser. Development does not require buying a
dividend-paying security, funding an account, waiting for a dividend, acquiring a
subscription, or accessing credentials/private reports.

This review read `src/data/ibkr_flex.py`, the closed source allowlist in
`src/operations/income.py`, `src/research/income.py`, the existing
[assessment](income-source-assessment.md) and
[acceptance record](income-source-acceptance.md), including its SAFE paper-capture
summary. No private XML was opened. The user's report that screenshots showed all
relevant fields selected is accepted as configuration context; the screenshots
were not independently inspected in this task. Do not ask the user to select the
same fields again or diagnose absent values on empty sections as a selection error.

## What official documentation establishes

Public official references below were retrieved on the review date. These are
field/report contracts, not evidence that this account experienced an event.

- [Cash Transactions](https://www.ibkrguides.com/reportingreference/reportguide/cash%20transactionsfq.htm)
  includes dividend cash payments and account, currency, security identity,
  date/time, amount, type and trade ID fields. An amount is a cash transaction
  amount; this reference does not specify a universal event-to-tax/fee join.
  In particular, its field list does not establish the semantics of every native
  `transactionID` or `actionID` encountered in XML. Preserve those attributes;
  require a documented mapping before using them as economic linkage.
- [Change in Dividend Accruals](https://www.ibkrguides.com/reportingreference/reportguide/change%20in%20dividend%20accrualsfq.htm)
  describes period accrual changes: `Po` adds an accrual; `Re` can mean correction,
  cancellation, or reversal on cash payout. Quantity is the quantity held before
  ex-date. Gross is rate times quantity; net subtracts tax and fee from gross.
  Change date and report date are separate fields. Those meanings support
  observation validation, not treating every `Re` as a cash payment or assuming
  the published accrual deductions were the final posted deductions.
- [Open Dividend Accruals](https://www.ibkrguides.com/reportingreference/reportguide/open%20dividend%20accrualsfq.htm)
  describes outstanding accruals intended for payment on pay date, with pre-ex
  quantity, gross rate/amount, net, tax, fee and account/security fields. It is a
  stock of outstanding accruals, not a history of paid cash. Pay date alone does
  not prove settlement; an empty open-accrual section does not exclude payments
  already made or accruals cancelled during the period.
- [Statement of Funds](https://www.ibkrguides.com/reportingreference/reportguide/statement%20of%20fundsfq.htm)
  is an optional period cash ledger with opening/closing balances and individual
  credits/debits, organized by currency. It documents a unique transaction ID and
  `DIV`, `FRTAX`, `PIL`, `CO` and `CA` activity codes for dividend, withholding,
  payment in lieu, correction and cancellation. This supports cash reconciliation;
  it does not specify that a correction ID identifies its predecessor or that a
  dividend and all deduction legs share one join key. Trade tax/commission
  fields cannot automatically be relabelled dividend tax/fee.
- [Flex Queries](https://ibkrguides.com/clientportal/performanceandstatements/flex.htm)
  permits field-level selection and chosen periods. Its stated historical
  availability is the current year and four prior calendar years; that is not
  evidence of point-in-time captures or amendment finality.
  [Activity query configuration](https://ibkrguides.com/clientportal/performanceandstatements/activityflex.htm)
  separately selects sections/fields, asset filters, accounts/models, output
  format, period and general settings. Thus all fields selected is necessary
  configuration evidence, but not the complete scope record.
  [Flex Web Service](https://ibkrguides.com/clientportal/performanceandstatements/flex-web-service.htm)
  retrieves predefined queries over HTTPS. Successful retrieval proves neither
  that every economic event was included nor that no later amendment will occur.

The reviewed references do not establish an intraday finality watermark,
universal correction/deduction linkage, or a complete effective-event feed.
This is a limitation of the evidence reviewed, not a claim that IBKR can never
supply further specifications. Reconcile a closed statement period only within
its documented scope; do not stretch its end date to the current cycle cutoff.

## Evidence available now

The documented paper milestone reports successful acquisition and local receipt
hash checks. It lists CashReport, StmtFunds, Trades, CorporateActions,
CashTransactions, both accrual sections and SecuritiesInfo; OpenPositions was
absent. Corporate actions and accrual sections were empty; cash observations had
no dividend-classified entry or nonempty action ID. This supports transport,
structure and absence of matching rows **in that returned report**, consistent
with a new account with no dividend history. It cannot exercise income linkage.

The reader preserves raw strings, decimals, leaf positions, absent/blank/zero,
summary/detail distinctions and actual acquisition clocks. Its receipt chain
verifies local consistency, explicitly not independent source authentication or
live-account mode. It creates zero canonical income records and always reports
blocked income admission. None of these facts is changed by selecting more fields.

Configuration evidence can now establish what was requested. Only the still
unrecorded scope needs preservation: exact saved query/version association,
account/model selection, period and timezone/date formats, detail/currency
settings, filters and acquisition time. Keep identifiers/configuration privately;
a safe review note can record which settings were checked. If these are already
present in the existing screenshots, reuse them. Query identity alone is not an
immutable configuration version, and no new query or capture is required for
development.

## Minimal remaining evidence, by acceptance claim

1. **A real posted-account dividend:** one original ordinary cash-dividend
   posting for one exact account/security/currency, its accrual/event identity,
   and all applicable deduction legs. Bind gross, quantity, rate, net and actual
   cash movement through documented business identifiers. Require explicit
   zero/applicability semantics, signs, rounding and FX where relevant. Reconcile
   the statement ledger without counting its summary or duplicate representation
   as another payment. A small sample validates that mapping, not general coverage.
2. **Lifecycle correctness:** a provider contract for revisions, corrections,
   cancellation and replay/overlap identity, including whether a later record is
   replacement or delta and how it names its predecessor. Preserve original and
   receipt clocks. Exercise each admitted case with source-backed examples when
   available; synthetic adversarial cases can test rejection now. If a lifecycle
   cannot be resolved, the affected claim stays unsupported; never silently drop
   it, match by description/date/amount, or infer a payout from `Re`.
3. **Complete absence or complete interval:** bind query configuration to the
   original delivered account/security period; establish provider semantics for
   empty results, delivery completeness, overlap/replay and late changes, with
   continuity/reconciliation evidence and a precise eligible cutoff. A complete
   statement cash ledger is narrower than complete dividend event terms. Even
   perfect field selection and balanced cash do not establish all required event
   capabilities or eliminate offsetting missing legs. No positive cash amount is
   needed to demonstrate a genuinely complete empty window, but no supported
   parser currently certifies that claim here.
4. **Pre-ex entitlement/terms:** timely original event/revision terms, applicable
   ex/record/pay conventions and actual account ownership/settlement evidence.
   The documented pre-ex quantity is useful broker accrual information; an end
   position or today's downloaded statement is not a pre-ex availability receipt.
   Resolve due bills, lending/payment-in-lieu and other exceptions or explicitly
   keep them outside the admitted ordinary-dividend profile. Separate terms
   authority is still needed where Flex does not establish this history.
5. **Local paper distributions and crypto:** a real credited broker account must
   never be reassigned to simulated holdings. Local paper needs an actual
   paid-per-unit occurrence tied to terms and a separately sourced
   `PaperDividendPolicy`; another holder's net-per-share is not transferable tax
   authority. Crypto needs its own exact instrument/custodian/protocol capability
   and interval evidence. These equity Flex sections cannot certify it.

Items 1–2 need actual posting evidence for positive broker-payment acceptance,
not as a prerequisite for writing software. Items 3–5 include semantic authority
that a populated report alone cannot fix. No amount of synthetic income grants
source acceptance. A future authorized original sample can close specific
observational gaps; it cannot substitute for the remaining contracts.

## Scope of the existing positive-source requirement

`_parsed_capture` admits only the two existing `operating-wire-2` dividend capture
kinds and independently parses their native Alpaca wire. It explicitly rejects
positive economics and complete coverage from that adapter. IBKR is not on this
allowlist; arbitrary `income_authorities` cannot extend it.

`verify_income_provenance` does **not** require nonzero income in each cycle. It
requires source-bound arrays and uninterrupted applicable coverage from the
authoritative account anchor to the actual cycle cutoff. For each non-crypto
instrument in snapshot bars/metadata it checks dividend terms plus either
account-specific payments (`posted-account`) or distributions (local paper).
Crypto hits a separate unsupported-custodial/protocol-authority rejection before
the equity coverage checks. Research similarly distinguishes crypto and requires
terms/distribution coverage over the economic observation interval.

Accordingly the requirement is correctly separated for the reviewed equity/crypto
scope: positive-source acceptance means sufficient authority, not compulsory
dividend income or treating crypto as dividend-paying equity. An equity-only
source cannot unblock a mixed cycle. No broad relaxation, no-staking shortcut,
operator-declared no-events waiver or normalized complete flag is justified.
The non-crypto branch is not a general certification of every possible asset
class, and this review does not validate other instruments or alter universe scope.

## Bounded recommendation and essential user actions

Proceed now with the existing diagnostic intake and tests. A useful future owned
increment is a **non-authorizing** configuration-to-observation review artifact:
bind the private saved-query snapshot and report hashes, identify requested versus
returned sections, and distinguish configuration checked, observations available,
and unresolved semantic authority. A checked configuration must never clear
coverage, payment-linkage or source-authentication gaps automatically. This note
does not implement that artifact or introduce another ledger.

Only propose a closed ordinary-dividend parser after the obligations above have
an evidenced mapping. Its acceptance tests must independently reparse original
bytes, reject altered clocks/accounts/amounts/IDs, duplicate or ambiguous revision
legs, summary-as-posting, unknown/partial coverage, and inferred tax or cash.
Preserve blocked mixed-crypto behavior until separate authority exists. Existing
rejection tests can be extended with labelled synthetic observations now; retain
fresh positive operating-cycle acceptance as an unmet integration requirement.

No immediate user action is essential to continue development. For later source
acceptance, preserve any missing configuration scope from the existing query.
If the reviewed public specifications remain insufficient, the precise provider
questions are: how dividend/accrual/deduction/revision identifiers relate; which
sign/zero/applicability rules apply; and what scope, late-amendment and cutoff
guarantees support an empty or complete delivery. Do not open a support request
automatically. Use a legitimately available original event sample later, with
separate authorization for private inspection; do not generate one by trading.

## Validation and execution boundary

Live orchestration reported `codex / gpt-6-astra`, `reasoningEffort=high`,
`serviceTier=default`, matching this child runtime. No additional threads were
spawned. No applicable `AGENTS.md` was found in checkout/ancestor searches;
repository `CLAUDE.md` was read. This change owns only this note; no source,
admission, dependency, configuration or sibling file is edited.

The existing suites already cover the relevant rejection boundaries, so no
redundant synthetic test file was added. Import assertions confirmed both
`src.data.ibkr_flex` and `src.operations.income` resolve under this checkout.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests/test_ibkr_flex_intake.py tests/test_operations_income_proof.py tests/test_action_coverage_accounts.py tests/test_research_income.py --disable-socket -q
```

Result: **57 passed in 1.88s**, no skips. These verify diagnostic and rejection
behavior, not provider acceptance or the retained positive-cycle requirements.
`git diff --check` passed. No secret/private-report access, authenticated provider
calls, trades, installations, containers, push, merge or PR was performed.
