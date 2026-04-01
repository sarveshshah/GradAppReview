# Document Naming Guide

Drop your `.txt` or `.md` files in this directory and run:

```bash
uv run python main.py --dir
```

## Naming Conventions

Files are auto-categorized by their name prefix. Use these prefixes:

| Prefix         | Category                  | Example                          |
|----------------|---------------------------|----------------------------------|
| `transcript_`  | Academic Transcript       | `transcript_upenn.txt`           |
| `transcript_`  | Academic Transcript       | `transcript_drexel.txt`          |
| `sop_`         | Statement of Purpose      | `sop.txt`                        |
| `statement_`   | Statement of Purpose      | `statement_of_purpose.txt`       |
| `resume_`      | Resume/CV                 | `resume.txt`                     |
| `cv_`          | Resume/CV                 | `cv.txt`                         |
| `linkedin_`    | LinkedIn Profile          | `linkedin.txt`                   |
| `website_`     | Personal Website/Portfolio| `website.txt`                    |
| `portfolio_`   | Personal Website/Portfolio| `portfolio.txt`                  |
| `essay_`       | Personal Essay            | `essay_diversity.txt`            |

## Tips

- You can have multiple transcripts: `transcript_upenn.txt`, `transcript_drexel.txt`
- For LinkedIn: export your profile or copy-paste from the LinkedIn PDF
- For your website: copy the key content (about, projects, experience sections)
- Any file not matching a prefix above will be labeled "Application Material"

## Configuration Files

These files customize the review for your target program:

| File | Purpose |
|------|---------|
| `program_context.txt` | Program details, curriculum, admissions criteria |
| `lors.txt` | Letter of recommendation summaries |
| `urls.txt` | Website URLs to scrape for portfolio content |

Copy the `.example.txt` templates to get started:
```bash
cp files/program_context.example.txt files/program_context.txt
cp files/lors.example.txt files/lors.txt
```

## What NOT to put here

- `.env` files or API keys
- Binary files (PDFs, images) — convert to text first
- This `NAMING.md` file and `.example.txt` files are tracked in git; everything else is gitignored
