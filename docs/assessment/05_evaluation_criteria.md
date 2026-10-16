# 5. Evaluation Criteria

> Source: `docs/Technical Assessment.pdf`, Section 5 (wording copied from the PDF)

## From the PDF
> We will assess your submission based on:
> - Computational geometry and mathematical reasoning.
> - Quality and consistency of the reconstructed roof.
> - Handling of noisy data, missing values, and obstructions.
> - Accuracy and geometric validity of solar panel placement.
> - Python code quality and algorithm design.
> - Validation, problem-solving, and technical decision-making.
> - Clarity in presenting and explaining your approach.
> - English fluency during the solution demonstration.
>
> **Important:** We value the quality of your reasoning and implementation over visual appearance
> alone. You should be prepared to explain and defend your technical decisions, demonstrate
> your implementation, and answer detailed questions about your approach during the technical
> discussion.
>
> Submit only work that you can explain and demonstrate yourself.

## What this means for how we work
| Criterion | How we address it |
|---|---|
| Geometry and maths reasoning | Use explicit math (plane fitting, normals, plane intersections, polygon ops) and document it |
| Roof quality and consistency | Report plane-fit error per face; faces must meet without gaps or steps |
| Noise, missing values, obstructions | Show before/after figures; state thresholds and why they were chosen |
| Panel accuracy and validity | Automated checks: every panel inside usable polygon, no overlaps, no obstruction contact |
| Code quality | Modular package, type hints, docstrings, config instead of magic numbers |
| Validation and decisions | Metrics, sanity checks, and a list of decisions with reasons |
| Clarity | Clear README, figures, and write-up |
| English fluency | The user's own delivery in the video |

**Rule:** Readable reasoning comes before pretty visuals. Keep every algorithm simple enough that the user can explain and defend it.
