# Income source assessment — 2026-10-27

No reviewed source currently proves all the income authorities required for a
fresh operating cycle. The source gate remains closed; canonical entitlement,
receivable, payment, fee and tax arithmetic does not establish provider coverage.
Actual authenticated delivery, subscription rights and prospective acceptance
have not been exercised. The retained positive cycle tests remain acceptance
requirements; they are not skipped or supplied fabricated coverage flags.

## Native Alpaca

The implemented capture and independent parser preserve corporate actions by
provider process date and activities by provider creation time. These endpoints
do not establish a complete effective-event window, pre-ex beneficial ownership,
or linked gross/net/all-deduction authority. Dates and net activity amounts cannot
be promoted into those claims. Empty responses produce explicit gaps, not proof
that no economic event occurred. See `src/data/adapters/dividends.py` and
`src/operations/income.py` for the closed source boundary.

## DTCC SR2025 component review

The exact SR2025 constrained XSDs and their message usage guides were inspected
locally. This was a format sufficiency review, not authenticated feed acceptance.
Schema packages remain outside the repository; no licensed schema bytes are
redistributed. References below use the guides' numbered pages.

| Authority | What the reviewed format establishes | Missing authority |
| --- | --- | --- |
| Event terms | CANO `seev.031.001.15`: event/security/option identities, ex/record/pay dates; option gross rate pp252–253 and cash-movement gross/net/tax/fee rates pp472–479 | Exact applicable revision/option, entitlement convention and settlement/due-bill applicability. Announcement rates are not posted account deductions. |
| Account movement | CACO `seev.036.001.16`: participant safekeeping account pp26–27, position balance, credit/debit, posting amount/date and optional dividend net amount pp69–70 | Authenticated participant authority and independent mapping to the actual trading account. Participant cash cannot be reassigned to a local simulated account. |
| Gross/fees/tax | CACO removes `GrssAmt` p70 and withholding/additional tax amounts pp77–78; retained `ChrgsAmt` is for reorganizations only pp80–81 | Account/event-linked gross and all deductions, affirmative zero/applicability, quantity, currency, rounding/FX and adjustment evidence. Gross minus net is not a decomposition. `SlctnFees` is an incentive/premium, not ordinary deduction proof. |
| Extensions | CACO `supl.011.001.11` pp16–19 includes movement association, payout/adjustment classification and optional amounts; tax-adjustment rate has specific legacy reason semantics | Exact core/extension association and original-versus-delta reconciliation. Missing fields do not mean zero. CANO `supl.001.001.17` p27 defaults an absent NRA-reportable flag to No only for that flag, not holder tax/fee applicability. |
| Business identity | Optional CACO `MvmntConfId` is a pagination-group identifier p15; extension `RDPRefNb` depends on event category. Instruction/customer references are not universal dividend payment IDs | Pinned BAH/delivery/replay profile, sender/recipient, original message identity, pagination, stable movement legs and duplicates. No BAH profile was established in these packages. Raw hashes alone do not deduplicate business payments. |
| Revisions/reversals | Guide pp16–22,26,64 describes replacements/cancellations, adjustment-only subsequent confirmations, same-day reversal/reallocation; CARE links the prior confirmation | Complete applicable CANO/CACN/CANO-E/CACO/CARE lifecycle and restart-safe correlation; separate original/posting clocks. Event-only deduplication would lose movements. |
| Complete window | Guide pp24–28 describes subscription, file/MQ schedule and holiday dependence; CACO is per participant/position type. CAPS pp17–18 removes Complete; Reconciled/Pending are per-event statuses | Authenticated account/instrument/capability scope, replay/start/end anchors, missing-message/file detection, counts/totals/continuity and explicit empty-window/final-watermark semantics. Last page, event completion or a message timestamp does not establish this. |

These are reviewed **component formats only**. No sufficient positive payment,
distribution or complete-coverage parser is registered from them. CANO and CANO-E
share a namespace/version but have different constrained schemas; matching the
namespace alone would be insufficient.

Official packages:

- [Entitlements and allocations SR2025](https://files.dtcc.com/download/assets/Entitlements+and+Allocations-2025.zip/25ce74a2b08d11f1a912fa5ff415476b)
- [Announcements SR2025](https://files.dtcc.com/download/assets/Announcements-2025.zip/20630d66b08d11f18bad1e308cb92e47)
- [Entitlements and allocations guide](https://files.dtcc.com/download/assets/ISO_20022_EntAlloc_UG.pdf/15b360f8369211f0a4da526112f6cc5f)
- [Release documentation hub](https://www.dtcc.com/products-and-services/asset-services/corporate-actions-processing/ca-documentation)

The hub reviewed on 2026-10-27 lists SR2025 deployed 2025-11-23 and SR2026
scheduled 2026-11-15. This review binds the following bytes, not a latest alias.

| Artifact | SHA-256 |
| --- | --- |
| Allocations ZIP | `2f328cbdfd6dbb8332e30aab221648791eb2ef2157808c132e45496c0bd4f5fb` |
| Announcements ZIP | `4738f2b3e4c6e08bc56885ef301b3ff9d3e9392404717396d5ab51d634e26092` |
| Allocations guide | `e1735faa3c90495a90b01b9057b615e91057e388b9fc66f0b23409fc4391578c` |
| CACO seev.036.001.16 XSD | `e8b2d8a3ec5dd13486febcc3f12a90258c1cb6780a70775e863a6da059837585` |
| CACO supl.011.001.11 XSD | `ae3253996f25a19ee1c7c85bd86d9b642d21ae3f5c87c95a9aa57b12ab64e191` |
| CANO seev.031.001.15 XSD | `ce504227291852e3597a0b57c61fc74912871c423f2fa0ad1f2e261842545a9d` |
| CANO supl.001.001.17 XSD | `c4fa9f5c40cf7ac6516ae45b086e278999ab2c0e29654a317cf8169984a8ebb5` |
| CANO-E seev.031.001.15 XSD | `14ed3bda9a53ac15608927f755980fb25fb5bf1c2e17a637bcdbd666e217496d` |
| CARE seev.037.001.16 XSD | `5de1e2653d1b0d253c968c0ed49a4ab0bcdeaa7150275145b3385a08d9c149df` |
| CAPS seev.032.001.09 XSD | `f5073e35e26dd2db6047f9e150d969c0c154fbe4d882c04edd8c003172359357` |

The next positive source implementation needs an identified, accessible provider
or statement format with actual account linkage, sufficient entitlement and
deduction authority, and complete scoped continuity evidence. Crypto protocol or
custodial action coverage remains a separate required authority.
