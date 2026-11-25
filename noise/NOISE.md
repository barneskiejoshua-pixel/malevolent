# BEFORE noise: base qwen3:4b-instruct-2507-q4_K_M, 4 runs

Benchmark: DRock's 60 execution contracts, each checked by a standard-library checker against the files the
agent left behind (never the agent's own claim). Ollama on an Intel Arc A380 (Vulkan), driver 31.0.101.4314,
every run passed preflight (model 100% on the GPU, no other runner on the card). 2026-11-24.

| Run | Pass / 60 | False successes | Task minutes |
|---|---|---|---|
| malev-noise-1 | 41 | 3 | 141 |
| malev-noise-2 | 41 | 4 | 115 |
| malev-noise-3 | 42 | 3 | 114 |
| malev-noise-4 | 38 | 4 | 133 |

Mean 40.5, SD 1.73, range 38-42.

Pairwise contracts that differ: runs 1 and 2: 10, runs 1 and 3: 5, runs 1 and 4: 7, runs 2 and 3: 7, runs 2 and 4: 5, runs 3 and 4: 6.

## Adoption floor (train/promote.py)

flips = 10 (the worst pair). ADOPT needs min(after pass) - max(before pass) > flips, so every AFTER run must
score at least 53 of 60, and max(after false successes) <= min(before) = 3.

## Per contract

P = pass, F = fail, S = false success (claimed done, checker says no). Held out = never trained on.

| Contract | Held out | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| code.add_function_and_test |  | F | F | F | F |
| code.cli_todo |  | F | F | F | F |
| code.fix_failing_test |  | P | P | P | P |
| code.project_scaffold | yes | F | F | F | F |
| code.script_run | yes | P | P | P | P |
| comp.github_to_summary |  | P | P | P | P |
| comp.pdf_to_sheet_chart | yes | F | F | F | F |
| comp.research_mla_verify |  | F | S | F | F |
| comp.search_to_report |  | P | F | F | F |
| comp.sheet_stats_to_slides | yes | P | P | P | P |
| excel.average |  | P | P | P | P |
| excel.chart | yes | F | F | F | F |
| excel.create_table |  | P | P | P | P |
| excel.currency |  | F | P | P | P |
| excel.filter |  | P | F | P | F |
| excel.modify_existing |  | P | F | F | F |
| excel.multisheet | yes | F | F | F | F |
| excel.percent |  | F | F | F | F |
| excel.sort | yes | P | F | P | F |
| excel.total_formula |  | P | P | P | P |
| fs.bulk_rename | yes | F | F | F | F |
| fs.copy |  | P | P | P | P |
| fs.create_dirs | yes | P | P | P | P |
| fs.delete |  | P | P | P | P |
| fs.duplicates |  | F | P | S | S |
| fs.move |  | P | P | P | P |
| fs.organize | yes | P | P | P | P |
| fs.rename |  | P | P | P | P |
| fs.search |  | P | P | P | P |
| fs.verify_deletion |  | P | P | P | P |
| pdf.extract_to_file | yes | P | P | P | P |
| pdf.read | yes | P | P | P | P |
| pptx.bullets |  | P | P | P | P |
| pptx.chart | yes | F | P | F | P |
| pptx.create |  | P | P | P | P |
| pptx.export_pdf |  | F | F | F | F |
| pptx.five_slides |  | P | P | P | P |
| pptx.four_titled |  | P | P | P | P |
| pptx.image | yes | S | S | S | S |
| pptx.modify |  | F | P | P | F |
| pptx.notes |  | P | P | P | P |
| pptx.reopen_verify | yes | P | P | P | P |
| web.download_image |  | P | P | P | P |
| web.download_pdf |  | P | P | P | P |
| web.github_search | yes | P | P | P | P |
| web.open_browser |  | P | P | P | P |
| web.open_url |  | P | P | P | P |
| web.readme |  | P | P | P | P |
| web.search_fact |  | S | S | P | S |
| web.trending_repo |  | P | P | P | P |
| word.apa_paper |  | S | P | S | S |
| word.export_pdf |  | P | P | P | P |
| word.font_global |  | P | P | P | P |
| word.insert_image |  | P | P | P | P |
| word.insert_table | yes | F | F | F | F |
| word.letter | yes | P | P | P | P |
| word.mla_essay |  | F | F | F | F |
| word.modify_existing | yes | P | P | P | P |
| word.report_headings |  | P | S | P | P |
| word.resume |  | P | P | P | F |

35 contracts pass in all four runs, 13 fail in all four, 12 flip.
