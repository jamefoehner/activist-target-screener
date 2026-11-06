# Labeling Guide: Activist Demands in 13D Item 4

Read this before labeling, and re-read it before pass 2. The labels are only as good as the consistency of these rules, and the rules matter more than any single judgment call.

## The one rule that matters most

**Label what the activist says it intends to do, not what it reserves the right to do.**

Almost every Item 4 contains boilerplate copied from the SEC form, a list of every possible action an investor might someday take:

> "...the Reporting Persons may engage in discussions with management... and may, from time to time, take actions that relate to or would result in any of the matters set forth in clauses (a) through (j) of Item 4 of Schedule 13D, including an extraordinary corporate transaction, a change in the board of directors..."

**Boilerplate like this is not a demand.** Ignore it. A demand needs a specific statement of intention, belief, or recommendation tied to this company, such as "intends to nominate," "believes the Issuer should explore a sale," "has requested two board seats," or "urges the Board to repurchase shares."

If the *only* content is boilerplate, plus statements like "acquired the shares because they were undervalued" or "intends to engage in discussions with management," label **engagement_only**.

## Demand categories

*capital_discipline was added after pass 1; earlier labels were reviewed and updated for it.*

Check every category that applies. More than one is common.

| Category | Check it when the filing says the activist intends, seeks, or recommends... | Do not check for... |
|---|---|---|
| **board_seats** | nominating directors, seeking board representation, or a settlement adding directors | boilerplate "change in the board" |
| **sale_or_review** | a sale of the company, a merger, a "strategic review" or "exploring strategic alternatives" | the activist's own possible acquisition offer (put that in notes) |
| **breakup_spin** | a spin-off, separation, or sale of a division or business unit | a sale of the whole company (that's sale_or_review) |
| **capital_return** | buybacks, special dividends, or returning cash to shareholders | general "improve capital allocation" with no return of capital mentioned |
| **operations_costs** | cost cuts, margin improvement, operational restructuring, or better execution | vague "enhance shareholder value" |
| **management_change** | replacing the CEO or other executives, or holding management accountable through leadership change | changes to the board only (that's board_seats or governance) |
| **governance** | declassifying the board, removing a poison pill, bylaw or charter changes, or shareholder rights | adding the activist's own nominees (that's board_seats) |
| **oppose_deal** | opposing an announced merger or acquisition, or demanding better terms in one | supporting a deal |
| **capital_discipline** | stopping or limiting acquisitions, cutting or re-evaluating capital spending, improving returns on capital, reducing debt, or improving capital allocation policy (phrased as a direction) | buybacks and dividends (that's capital_return); bare topics like "capital allocation" |
| **engagement_only** | none of the above: just "investment purposes" and "discussions with management" | — never check this together with another category |

## Filings that aren't activist campaigns

Some 13Ds come from a fund that *controls* the company, for example after spinning off or taking public its own subsidiary. These describe ownership arrangements, not demands. Don't label demands. Instead:
- check **ambiguous**
- start the notes with **"EXCLUDE: not a campaign"** plus the reason (e.g., "controlling parent after IPO")
- leave every demand unchecked

These items are dropped from the evaluation, and their accession numbers go into `data/exclusions.csv`, so the event study and model drop them too.

## Other fields

- **Board seats requested.** Enter the number explicitly stated: nominated, requested, or agreed in a settlement. Enter 0 if no number is given, even if board representation is sought. "A minority slate" with no number counts as 0. Note it.
- **Proxy fight threatened.** Check yes only if the filing says the activist has nominated, intends to nominate, or may solicit proxies or call a special meeting. Boilerplate about possibly soliciting proxies doesn't count.
- **Tone:**
  - **cooperative:** describes a constructive dialogue, a settlement, or support for management
  - **neutral:** factual, with no criticism (the default)
  - **confrontational:** criticizes management or the board, cites underperformance, or threatens action
- **Ambiguous.** Flag filings where you genuinely could argue either way after applying these rules. Use it sparingly. These items get reviewed, and they help explain model errors later.
- **Notes.** Record the phrase that decided any close call. That makes pass 2 and the error analysis much easier.

## Edge cases

- **A letter attached as an exhibit.** If Item 4 says "see the letter attached as Exhibit 99.1" and summarizes it, label from the summary in Item 4 only. The model sees only Item 4, so the label must too.
- **Past actions.** "Has delivered a letter urging a sale" counts as a demand (sale_or_review). Activists describe what they've already asked for.
- **Conditional language.** "If the Board does not act, the Reporting Persons intend to nominate..." means **board_seats** checked and **proxy fight = yes**.
- **"Maximize value" or "unlock value" with no specific means:** treat as engagement_only, unless a specific route is named.
- **Proxy fights against a deal.** Soliciting proxies *against* a transaction, such as voting down a merger at a special meeting, counts as **proxy fight = yes**, even with no directors nominated.
- **Procedural requests tied to a deal fight,** such as moving a record date or meeting date: don't check governance. They're tactics in the oppose_deal fight. Mention them in notes.
- **Disclosure complaints and criticism of the board** ("failure to disclose," "strongly oppose") mean **tone = confrontational**.
- **Existing board seats.** A seat the activist already holds through a prior agreement (for example, an Investor Rights Agreement) is not a board_seats demand. Check board_seats only if this filing describes seeking seats or a new settlement granting them.
- **Independent directors vs. the activist's own nominees.** Asking the company to add *independent* directors, not the activist's own people, counts as **governance**, not board_seats.
- **"Strategic alternatives."** The phrase counts as **sale_or_review** only inside a stated intention or plan, such as "intend to discuss… which may include the exploration of strategic alternatives." Inside reserve-the-right language ("reserve the right…," "may take positions or make proposals with respect to… strategic alternatives"), it's boilerplate. If it comes with a specific example, also check that example's category when it clearly fits. If the example is vague (for example, "a restructuring of its key business segments"), don't check breakup_spin. Flag it as ambiguous and quote the phrase in notes.
- **Executive pay.** Aligning management compensation with performance counts as **governance**, not management_change.
- **Hedged but company-specific demands.** If a demand is written specifically for this company, even with softening words like "potentially" or "may seek," check the category. Flag it as ambiguous, and quote the hedge in notes. The boilerplate to ignore is the generic list of actions copied from the SEC form, not every sentence that uses "may."
- **Topic nouns vs. direction verbs.** Within a list of discussion items, a bare noun ("operating performance," "capital allocation," "board composition") is a topic, so don't check it. An item phrased as a change ("**improving** operating performance," "**providing for** a significant return of capital," "**exploring** a sale," "**adding** directors") names a direction, so check it. Flag it ambiguous when the verb is generic ("improving").
- **Judge sentences, not paragraphs.** A boilerplate paragraph can contain one real sentence, such as "have encouraged the Issuer to undertake a strategic review process" in the middle of standard "may take positions…" language. Read every sentence. What it states decides the label, not the paragraph around it.
- **Lists of discussion topics.** "Intend to engage in discussions… that may relate to the business, management, operations, governance and board composition…" lists subjects, not demands, even after "intend" and even with a detail like "(including cost structure)." Label engagement_only. A demand names a direction (cut costs, add directors, sell a division), not just a topic.
- **Nominee counts that change.** Enter the number of directors actually nominated in the filing. If the filing also says some will be withdrawn, note the revised number in notes ("6 nominated; intends to withdraw 2 → 4") and flag it ambiguous. When scoring, accept either number for these items.
- **Legal disclaimers don't override substance.** Phrases like "is not taking any position or making any recommendation" are common legal hedges. If the filing argues a deal undervalues the company and seeks a better offer, check oppose_deal anyway. Flag it ambiguous and quote the disclaimer in notes.
- **A strategic review the company already started.** Planning to discuss the company's *ongoing* review isn't a sale_or_review demand, because the activist isn't asking for anything new. Check it only if the activist states a position on the review, such as urging a sale or a particular outcome. Flag it ambiguous and note it.
- **Selling divisions vs. selling the company.** A push to sell or separate individual divisions is **breakup_spin**, even when the alternative on the table is a sale of the whole company. Check sale_or_review only when the activist itself is pushing for a sale or review of the whole company.
- **Tone when signals conflict.** If the activist cites weak results but explicitly frames its engagement as constructive or collaborative, with no criticism of specific people and no threats, use **cooperative**. Use **confrontational** only for direct criticism of management or the board, or threats such as proxy fights or public campaigns.

## Pass 2

About a week after finishing pass 1, re-read this guide and label the 20 blind items in the app's pass 2. Don't look at your pass-1 labels. The agreement between passes sets the ceiling for judging the models. If you agree with yourself 85% of the time on a category, a model scoring 85% there is roughly at human level.
