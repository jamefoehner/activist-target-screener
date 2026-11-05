# Can Language Models Read Activist Filings? An Evaluation of Demand Extraction from SEC 13Ds

**James Foehner** · Part of the [activist target screener](../README.md) project · 2026

## Summary

When an activist hedge fund crosses 5% ownership, it files a Schedule 13D whose Item 4 ("Purpose of Transaction") describes what it wants. The text is long, legal, and deliberately hedged, and most of it is boilerplate copied from the SEC form. I built a benchmark to test whether language models can tell real demands from boilerplate.

I hand-labeled 100 initial 13Ds across ten demand categories, seat counts, proxy-fight threats and tone, writing a labeling guide as I went. A second person labeled 20 of the filings independently from the guide alone, and I re-labeled the same 20 a week later. I then ran Claude Haiku 5.5 and Claude Sonnet 5.5 with three prompts on the 86 campaign filings not used as worked examples.

- **With the labeling guide, Claude Sonnet 5.5 got every demand category right on 83% of filings** (micro-F1 0.92), and on 98% of the filings I had not flagged as ambiguous.
- **On the 20 filings labeled by both humans, Sonnet with the guide matched the reference labels on all 20**, versus 18 of 20 for the second human.
- **The written rules mattered more than model size.** Adding the guide raised Haiku from 74% to 80% of filings fully correct; switching from Haiku to Sonnet with the same prompt added about 2 points, at roughly 20 times the cost.
- **Almost all remaining errors are judgment calls.** Every system scored at least 95% on clear filings; on filings I had flagged as ambiguous, the best system scored 69%.
- **The models audited my labels.** Filings where every model disagreed with me surfaced six labeling errors, each confirmed against the written guide.

## The task

Each model reads one filing's Item 4 and returns JSON with the same fields I labeled:

| Field | Values |
|---|---|
| Demand categories (check all that apply) | board seats · sale or strategic review · breakup or spin-off · capital return · operations and costs · management change · governance · oppose a pending deal · capital discipline · engagement only (no specific demand) |
| Seats requested | integer, 0 if no number stated |
| Proxy fight threatened | yes / no |
| Tone | cooperative / neutral / confrontational |

The core difficulty is the rule at the top of the guide: **label what the activist says it intends to do, not what it reserves the right to do.** Nearly every filing lists every possible action ("may propose… a merger… changes to the board… dividend policy…"). Those lists are not demands, but they contain every keyword a careless reader would match on.

## Data and labels

- **Filings.** 100 initial 13Ds sampled (stratified by activist) from 232 campaigns by eight activists, 2010–2026, collected from SEC EDGAR. Item 4 was extracted from plain-text, HTML and XML filings.
- **Labeling guide.** Written during labeling, with rules for hedged demands, discussion-topic lists versus stated directions, settlements, legal disclaimers, nominee counts and tone conflicts. It is the same document given to the second labeler and to the models.
- **Non-campaigns.** Labeling surfaced 13Ds that were not new campaigns (shares inherited in spin-offs, control positions, restructurings, amendment text filed as an original). These were excluded from the evaluation, and an automated scan built from them removed 30 of 262 filings from the wider dataset.
- **Human agreement.** On 20 shared filings, the second labeler and my own re-label a week later both matched my original labels on every demand category; the only disagreement was a missed nomination in one long filing (proxy fight and seat count).

## Models and prompts

| Prompt | What the model sees |
|---|---|
| minimal | one-line definitions of each category |
| guide | the full labeling guide (~2,900 tokens) |
| guide + examples | the guide plus three worked examples: a boilerplate-only filing, a proxy fight, and a multi-demand filing |

The three example filings are excluded from scoring for every prompt, so all prompts are scored on the same 86 filings. Neither model accepts temperature 0, so both ran at default sampling.

## Results

### Overall (86 filings, final reference labels)

| System | Micro-F1 | All demands correct | Clear filings | Ambiguous filings | Tone |
|---|---|---|---|---|---|
| Haiku 5.5, minimal | 0.87 | 74% | 95% | 56% | 78% |
| Haiku 5.5, guide | 0.90 | 80% | 95% | 67% | 90% |
| Haiku 5.5, guide + examples | 0.91 | 80% | 98% | 64% | **94%** |
| Sonnet 5.5, minimal | 0.89 | 79% | **100%** | 60% | 81% |
| **Sonnet 5.5, guide** | **0.92** | **83%** | 98% | **69%** | 86% |
| Sonnet 5.5, guide + examples | 0.92 | 81% | 98% | 67% | 92% |

Seat counts were exact on 97–99% of filings and proxy-fight calls correct on 98–100% for every system.

### Humans and models on the same 20 filings

| System | All demands correct | Ambiguous filings | Tone |
|---|---|---|---|
| Second human labeler | 90% | 100% | 100% |
| My own re-label, one week later | 90% | 100% | 100% |
| Haiku 5.5, guide | 90% | 89% | 85% |
| **Sonnet 5.5, guide** | **100%** | **100%** | 90% |

### By category (F1, Sonnet 5.5 with the guide)

| Category | F1 | Filings labeled positive |
|---|---|---|
| Engagement only | 0.95 | 39 |
| Board seats | 0.93 | 23 |
| Operations and costs | 0.97 | 16 |
| Sale or review | 0.88 | 18 |
| Capital discipline | 0.88 | 8 |
| Breakup or spin-off | 0.86 | 8 |
| Governance | 0.80 | 9 |
| Capital return / oppose deal | 1.00 | 4 each |

### What the results show

1. **Rules beat scale.** The guide improved both models more than moving from Haiku to Sonnet did. A small model with precise instructions comes close to a large one.
2. **Examples help with tone, not demands.** Worked examples raised tone accuracy for both models (Haiku from 90% to 94%) but did not improve demand extraction, and slightly lowered it on ambiguous filings.
3. **Clear filings are essentially solved.** The gap between systems is almost entirely on the filings a human also found hard.

## Error analysis

Most disagreements are concentrated on filings I had flagged as ambiguous (84% before label corrections). Reading them, they fall into a few types:

| Error type | What happens |
|---|---|
| **Different strictness on hedged signals** | Models tend to leave board seats unchecked and default to "engagement only" on hedged phrasing such as "potentially seeking board representation." My guide counts these as hedged demands; models read them more strictly. |
| **Tone calibration** | The largest error source. Models label proxy-fight filings "neutral" where the guide says a nomination threat makes a filing confrontational, and label polite template language "cooperative" where I said neutral. |
| **Losing track of reversals** | In settlement filings (New Relic, Southwest Gas), models reported a proxy fight even though the nomination had been withdrawn under the agreement. |
| **Direction versus topic** | Models sometimes check operations or capital discipline on bare topics ("operational opportunities," "capital allocation") where the guide requires a stated direction. |

The pattern matters: most model "errors" reflect a different threshold for judgment calls, not misreading. The genuine reading failures are concentrated in long settlement filings where earlier actions were reversed.

## The models audited the labels

On nine filing–field pairs, all six model and prompt combinations disagreed with my label. Rereading each against the guide, six were my errors:

- **Board seats on Triple Flag (Elliott):** the activist's employee already sat on the board under an existing agreement, which the guide says is not a demand.
- **Tone on an Elliott template:** "constructive dialogue" maps to cooperative under the guide.
- **Capital discipline on four filings:** checked during a later review on bare "capital allocation" topics, contrary to the rule.

Two of the nine stayed as labeled (a hedged board signal at Liberty Media, and a seat count that is a documented soft field), and one is genuinely arguable. I corrected only labels the written guide clearly contradicted, and report results before and after:

| Sonnet 5.5 with the guide | Original labels | Corrected labels |
|---|---|---|
| Micro-F1 | 0.90 | 0.92 |
| All demands correct | 80% | 83% |
| Second human, all demands correct (20 filings) | 95% | 90% |

Two lessons follow. Unanimous model disagreement is a cheap way to find labeling errors: it flagged a 7% error rate in my reference labels. And two humans agreeing does not make a label right: on one of the 20 shared filings, the second labeler and my week-later re-label had both matched my original, incorrect label, which is why their agreement fell from 95% to 90% after correction.

## Cost

| | Input tokens (all 3 prompts) | Approximate cost | Seconds per filing |
|---|---|---|---|
| Haiku 5.5 | ~1.3 million | ~$0.20 | 2.5 |
| Sonnet 5.5 | ~1.3 million | ~$3.00 | 2.1 |

No reply from either model failed to parse as valid JSON.

## Limitations

- **The guide was written on this sample.** Its rules were shaped by these exact filings, which favors both the second human and the models. Scores on unseen filings are likely lower; a fresh labeled set is the clearest next step.
- **Small samples.** 86 scored filings, and only 20 with multiple human labels, so differences of a few points between systems are not significant.
- **Labels corrected after seeing model output.** Corrections followed the written guide and both versions are reported, but adjudication toward model answers flatters the corrected scores.
- **Item 4 only.** Many activists put specific demands in attached letters (Exhibit 99.1) rather than Item 4, so the benchmark measures what Item 4 says, not everything the activist demanded.
- **Default sampling.** Neither model accepts temperature 0, so reruns can vary slightly.
- **Two models from one provider.** An open-source model would make the comparison broader.

## Next steps

- Label 20–30 filings outside the sample to measure performance on text the guide was not written from.
- Extract attached activist letters and test whether demand coverage improves.
- Add an open-source model, and rerun each model twice to measure run-to-run consistency.
- Use the extracted demands to test which kinds of demands drive the stock reactions measured in the event study.

## Reproducing

```bash
python3 src/extract_item4.py --user-agent "Name email@example.com"   # Item 4 text + labeling sample
streamlit run src/label_app.py                                        # hand-labeling (pass 1, pass 2, second labeler)
python3 src/compare_labels.py                                          # human agreement
export ANTHROPIC_API_KEY=...
python3 src/run_eval.py --models claude-haiku-5-5 claude-sonnet-5-5 \
    --prompts minimal guide guide_examples --example-ids 1 3 5
```

The labeling guide is in `docs/LABELING_GUIDE.md`; labels are in `data/labels_pass1.csv`, `labels_pass2.csv` and `labels_passB.csv`.
