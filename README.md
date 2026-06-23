# farhanzafrani.github.io

Personal portfolio and blog — pure HTML/CSS/JS, generated from YAML data files.

## Prerequisites

- Python 3.8+
- pip

## Setup

```bash
pip install -r requirements.txt
```

## Build

Regenerates `index.html`, `blog.html`, and all `blog/<slug>.html` pages from the data files.

```bash
python scripts/build_portfolio.py
```

**Data sources:**

| File | Controls |
|------|----------|
| `data/portfolio.yml` | About, experience, projects, skills, education, contact |
| `data/blogs.yml` | Blog post metadata (title, date, tags, slug) |
| `content/blogs/<slug>.md` | Blog post body (Markdown) |

## Build + Serve

Build and immediately serve the site locally on port 8000:

```bash
python scripts/build_portfolio.py -s
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Deploy

Push to the `master` branch — GitHub Pages serves the site automatically from the repo root.

```bash
git add .
git commit -m "your message"
git push origin master
```
