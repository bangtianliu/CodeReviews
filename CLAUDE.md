# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains automated analyses of code reviews collected using Claude. It organizes detailed review comments and generates actionable code style checklists from two projects:

- **iree/** - Reviews from [IREE](https://github.com/iree-org/iree) compiler infrastructure (1372 comments, 103 PRs)
- **tuner/** - Reviews from [AMD Shark Tuner](https://github.com/nod-ai/amd-shark-ai) performance tuning (684 comments, 72 PRs)

Each project directory contains:
- `detailed_reviews.md` - Full review comments with context
- `checklist_from_reviews.md` - Actionable checklists derived from review patterns

## Common Tasks

### Generating Review Data

Using Claude Code with GitHub CLI:
1. Fetch all PRs from the repository
2. Filter for PRs reviewed by specific reviewers
3. Retrieve detailed review comments via GitHub API
4. Analyze patterns and generate documentation

### Using the Data

Before submitting new PRs to IREE or the Tuner:
1. Review the relevant `checklist_from_reviews.md` for coding standards
2. Search `detailed_reviews.md` for similar patterns
3. Ensure code meets quality standards identified through reviews

## Key Coding Standards (from Reviews)

### IREE (C++/MLIR)
- Use `auto` only when type is obvious from RHS; use `const auto&` in range loops
- Prefer early returns to reduce nesting
- Use `llvm::cast/dyn_cast/isa` (not C-style casts); don't assert after `llvm::cast`
- Pass vectors as `ArrayRef` instead of by value
- Use Builder convenience methods for attribute creation
- Mark helper functions as `static`
- Remove debug prints before merging

### Tuner (Python)
- Start function names with active verbs
- Use direct boolean assertions, not `== True/False`
- Avoid `Any` type annotation
- Compare lists directly in tests
- Hoist invariant checks outside loops
- Don't guard for loops with if (they handle empty lists fine)
