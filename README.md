# Code Review Analysis Collection

This repository contains automated analyses of code reviews I've received, collected using Claude to help guide my future development and improve code quality.

## Purpose

I use Claude to:
- Collect and organize all detailed code review comments from my pull requests
- Identify patterns and common feedback themes across reviews
- Generate actionable code style checklists based on actual reviewer feedback
- Learn from past reviews to improve future submissions

## Structure

### `iree/`
Code reviews from the IREE project:
- **detailed_reviews.md** - All review comments with full context (640 comments across 74 PRs)
- **checklist_from_reviews.md** - Actionable checklist derived from review patterns

## How to Use

Before submitting new PRs:
1. Review the relevant code style checklist for the project
2. Search detailed reviews for similar patterns or solutions
3. Learn from past feedback to avoid repeating mistakes
4. Ensure code meets quality standards identified through reviews

## How This is Generated

Using Claude Code with GitHub CLI:
1. Fetch all PRs from the repository
2. Filter for PRs reviewed by specific reviewers
3. Retrieve detailed review comments via GitHub API
4. Analyze patterns and generate documentation

---

*This is a living document that grows as I receive more code reviews.*
