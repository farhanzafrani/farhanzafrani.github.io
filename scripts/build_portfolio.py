#!/usr/bin/env python3
"""
Portfolio Build Script
======================
Reads  data/portfolio.yml  and  data/blogs.yml  and writes:
  index.html, blog.html, blog/<slug>.html

Usage
-----
    python scripts/build_portfolio.py          # generate all pages
    python scripts/build_portfolio.py -s       # generate + serve on :8000
"""

import sys
import json
import os
import re
import html as _html
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  pip install pyyaml")

ROOT        = Path(__file__).resolve().parent.parent
DATA        = ROOT / "data" / "portfolio.yml"
BLOGS       = ROOT / "data" / "blogs.yml"
CONTENT_DIR = ROOT / "content" / "blogs"
OUT         = ROOT / "index.html"
BLOG_OUT    = ROOT / "blog.html"
BLOG_DIR    = ROOT / "blog"


# ── Utilities ──────────────────────────────────────────────────────

def e(s: object) -> str:
    """HTML-escape a value."""
    return _html.escape(str(s))


# ── Markdown → HTML ────────────────────────────────────────────────

def _fmt(text: str) -> str:
    """Escape and apply bold/italic/code/link to already-safe text segments."""
    text = _html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'`([^`]+)`',     lambda m: f'<code>{m.group(1)}</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: f'<a href="{_html.escape(m.group(2))}" target="_blank" rel="noopener">{_html.escape(m.group(1))}</a>',
                  text)
    return text


def _inline_md(text: str) -> str:
    """Convert inline markdown to HTML, preserving image tags before escaping."""
    img_re = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    parts = []
    last = 0
    for m in img_re.finditer(text):
        parts.append(_fmt(text[last:m.start()]))
        alt = _html.escape(m.group(1))
        src = _html.escape(m.group(2).strip())
        parts.append(f'<img src="{src}" alt="{alt}" loading="lazy" class="inline-img">')
        last = m.end()
    parts.append(_fmt(text[last:]))
    return ''.join(parts)


def _render_table(lines: list) -> str:
    rows = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\|[\s\-|:]+\|$', stripped):
            continue
        cells = [c.strip() for c in stripped.strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return ''
    html = '<div class="table-wrap"><table>\n<thead><tr>'
    for cell in rows[0]:
        html += f'<th>{_inline_md(cell)}</th>'
    html += '</tr></thead>\n'
    if len(rows) > 1:
        html += '<tbody>\n'
        for row in rows[1:]:
            html += '<tr>' + ''.join(f'<td>{_inline_md(c)}</td>' for c in row) + '</tr>\n'
        html += '</tbody>\n'
    html += '</table></div>'
    return html


_YT_RE = re.compile(
    r'^https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^&]*&)*v=|youtu\.be/)([\w-]+)')

def _youtube_embed(url: str) -> str:
    m = _YT_RE.match(url)
    if not m:
        return f'<p><a href="{_html.escape(url)}" target="_blank" rel="noopener">{_html.escape(url)}</a></p>'
    vid = m.group(1)
    return (f'<div class="video-embed">'
            f'<iframe src="https://www.youtube.com/embed/{vid}" '
            f'title="YouTube video" frameborder="0" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
            f'allowfullscreen loading="lazy"></iframe>'
            f'</div>')


def md_to_html(text: str) -> str:
    """Convert markdown to HTML. Supports headings, lists, code blocks, tables, images, videos."""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code = _html.escape('\n'.join(code_lines))
            lang_cls = f' class="language-{e(lang)}"' if lang else ''
            out.append(f'<pre><code{lang_cls}>{code}</code></pre>')

        # Headings
        elif stripped.startswith('### '):
            out.append(f'<h3>{_inline_md(stripped[4:])}</h3>')
        elif stripped.startswith('## '):
            out.append(f'<h2>{_inline_md(stripped[3:])}</h2>')
        elif stripped.startswith('# '):
            out.append(f'<h1>{_inline_md(stripped[2:])}</h1>')

        # Block image: ![alt](src) on its own line → <figure>
        elif re.match(r'^!\[', stripped) and re.match(r'^!\[[^\]]*\]\([^)]+\)\s*$', stripped):
            m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', stripped)
            alt = _html.escape(m.group(1))
            src = _html.escape(m.group(2).strip())
            caption = f'<figcaption>{alt}</figcaption>' if alt else ''
            out.append(f'<figure class="blog-figure"><img src="{src}" alt="{alt}" loading="lazy">{caption}</figure>')

        # Standalone YouTube URL on its own line → embed
        elif _YT_RE.match(stripped):
            out.append(_youtube_embed(stripped))

        # Table (consume all consecutive | lines)
        elif stripped.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            out.append(_render_table(table_lines))
            continue

        # Unordered list
        elif re.match(r'^[-*]\s', stripped):
            items = []
            while i < len(lines) and re.match(r'^[-*]\s', lines[i].strip()):
                items.append(f'<li>{_inline_md(lines[i].strip()[2:])}</li>')
                i += 1
            out.append('<ul>\n' + '\n'.join(items) + '\n</ul>')
            continue

        # Ordered list
        elif re.match(r'^\d+\.\s', stripped):
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i].strip()):
                text_part = re.sub(r'^\d+\.\s', '', lines[i].strip())
                items.append(f'<li>{_inline_md(text_part)}</li>')
                i += 1
            out.append('<ol>\n' + '\n'.join(items) + '\n</ol>')
            continue

        # Horizontal rule
        elif re.match(r'^---+$', stripped) or re.match(r'^\*\*\*+$', stripped):
            out.append('<hr>')

        # Empty line
        elif not stripped:
            out.append('')

        # Regular paragraph
        else:
            out.append(f'<p>{_inline_md(stripped)}</p>')

        i += 1

    return '\n'.join(out)


# ── Markdown file loading ──────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple:
    """Return (meta_dict, body_str) from a markdown file with optional YAML front matter."""
    if not text.startswith('---'):
        return {}, text
    end = text.find('\n---', 3)
    if end == -1:
        return {}, text
    meta = yaml.safe_load(text[3:end]) or {}
    body = text[end + 4:].lstrip('\n')
    return meta, body


def load_markdown_posts() -> list:
    """Load posts from content/blogs/*.md (YAML front matter + markdown body)."""
    if not CONTENT_DIR.exists():
        return []
    posts = []
    for md_file in sorted(CONTENT_DIR.glob('*.md')):
        text = md_file.read_text(encoding='utf-8')
        meta, body = parse_frontmatter(text)
        if not meta.get('slug'):
            meta['slug'] = md_file.stem
        meta['content'] = body
        posts.append(meta)
    return posts


# ── Shared partials ────────────────────────────────────────────────

def _head(title: str, description: str, css: str = 'assets/css/portfolio.css') -> str:
    return f"""<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <meta name="description" content="{description}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&amp;family=JetBrains+Mono:wght@300;400;500;700&amp;display=swap"
        rel="stylesheet">
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
        crossorigin="anonymous">
  <link rel="stylesheet" href="{css}">
</head>"""


def _bg() -> str:
    return """  <div class="bg-orbs" aria-hidden="true">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
  </div>
  <div class="bg-grid" aria-hidden="true"></div>"""


def _navbar(p: dict, *, active: str = '', prefix: str = '', on_index: bool = False) -> str:
    home_href = '#hero' if on_index else f'{prefix}index.html'
    sections = [
        ('about',      'About'),
        ('experience', 'Experience'),
        ('projects',   'Projects'),
        ('skills',     'Skills'),
        ('education',  'Education'),
        ('contact',    'Contact'),
    ]
    li = ''
    for sid, label in sections:
        href = f'#{sid}' if on_index else f'{prefix}index.html#{sid}'
        cls  = ' class="active"' if active == sid else ''
        li  += f'<li><a href="{href}"{cls}>{label}</a></li>'
    blog_href = f'{prefix}blog.html'
    blog_cls  = ' class="active"' if active == 'blog' else ''
    li += f'<li><a href="{blog_href}"{blog_cls}>Blog</a></li>'
    return f"""  <nav class="pf-nav" role="navigation" aria-label="Main navigation">
    <div class="pf-nav__inner">
      <a class="pf-nav__logo" href="{home_href}" aria-label="Home">MF</a>
      <ul class="pf-nav__links" id="nav-links">
        {li}
      </ul>
      <button class="pf-nav__burger" aria-label="Toggle menu"
              aria-expanded="false" aria-controls="nav-links">
        <span></span><span></span><span></span>
      </button>
    </div>
  </nav>"""


def _footer(p: dict) -> str:
    return f"""  <footer class="pf-footer">
    <div class="container">
      <p>Built with HTML, CSS &amp; Python &middot;
         <a href="https://github.com/{e(p['github'])}/{e(p['github'])}.github.io"
            target="_blank" rel="noopener">Source</a>
         &middot; &copy; 2025 {e(p['first_name'])} {e(p['last_name'])}
      </p>
    </div>
  </footer>"""


# ── Section builders ───────────────────────────────────────────────

def _hero(d: dict) -> str:
    p, h = d["personal"], d["hero"]

    stats = "".join(
        ('  <div class="stat__div"></div>\n' if i else "") +
        f'  <div class="stat"><span class="stat__n">{e(s["num"])}</span>'
        f'<span class="stat__l">{e(s["label"])}</span></div>\n'
        for i, s in enumerate(h["stats"])
    )
    badges = "".join(
        f'<span class="hero__badge hb-{e(b["color"])}">{e(b["text"])}</span>'
        for b in h["badges"]
    )
    return f"""
  <!-- HERO -->
  <section id="hero" class="hero">
    <div class="hero__content">
      <div class="hero__text">
        <div class="hero__tag">AI &amp; Automation Engineer</div>
        <h1 class="hero__name">
          <span>{e(p['first_name'])}</span><br>
          <span class="gradient-text">{e(p['last_name'])}</span>
        </h1>
        <p class="hero__subtitle">
          <span class="typed-text"></span><span class="typed-cursor" aria-hidden="true">|</span>
        </p>
        <div class="hero__cta">
          <a href="#projects" class="btn btn-primary">View Projects</a>
          <a href="https://linkedin.com/in/{e(p['linkedin'])}" class="btn btn-outline"
             target="_blank" rel="noopener"><i class="fab fa-linkedin"></i>&nbsp;LinkedIn</a>
        </div>
        <div class="hero__stats">
{stats}        </div>
      </div>
      <div class="hero__photo">
        <div class="hero__ring">
          <div class="hero__frame">
            <img src="assets/img/profile_pic.png"
                 alt="{e(p['first_name'])} {e(p['last_name'])}" loading="eager">
          </div>
        </div>
        <div class="hero__badges">{badges}</div>
      </div>
    </div>
    <div class="hero__scroll" aria-hidden="true">
      <span>scroll</span><div class="scroll-bar"></div>
    </div>
  </section>"""


def _about(d: dict) -> str:
    p, ab = d["personal"], d["about"]
    paras = "\n".join(f"          <p>{para.strip()}</p>" for para in ab["paragraphs"])
    tags  = "\n".join(
        f'          <span class="tag t-{e(t["color"])}">{e(t["text"])}</span>'
        for t in ab["interests"]
    )
    return f"""
  <!-- ABOUT -->
  <section id="about" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">01 / about</span>
        <h2>Who I Am</h2>
      </div>
      <div class="about-grid">
        <div class="about-text">
{paras}
          <span class="tag-label">Research &amp; Engineering Interests</span>
          <div class="tag-group">
{tags}
          </div>
        </div>
        <div class="about-info">
          <div class="info-row">
            <i class="fas fa-map-marker-alt"></i>
            <div><div class="info-label">Location</div>
                 <div class="info-value">{e(p['location'])}</div></div>
          </div>
          <div class="info-row">
            <i class="fas fa-building"></i>
            <div><div class="info-label">Company</div>
                 <div class="info-value">
                   <a href="{e(p['company_url'])}" target="_blank" rel="noopener">{e(p['company'])}</a>
                 </div></div>
          </div>
          <div class="info-row">
            <i class="fas fa-graduation-cap"></i>
            <div><div class="info-label">Education</div>
                 <div class="info-value">{e(p['education_short'])}</div></div>
          </div>
          <div class="info-row">
            <i class="fas fa-envelope"></i>
            <div><div class="info-label">Email</div>
                 <div class="info-value">
                   <a href="mailto:{e(p['email'])}">{e(p['email'])}</a>
                 </div></div>
          </div>
          <div class="info-row">
            <i class="fas fa-language"></i>
            <div><div class="info-label">Languages</div>
                 <div class="info-value">{e(p['languages'])}</div></div>
          </div>
        </div>
      </div>
    </div>
  </section>"""


def _experience(d: dict) -> str:
    items = ""
    for exp in d["experience"]:
        pts = "\n".join(f"              <li>{e(pt)}</li>" for pt in exp["points"])
        items += f"""
      <div class="tl-item">
        <div class="tl-card">
          <div class="tl-year">{e(exp['year'])}</div>
          <div class="tl-role">{e(exp['role'])}</div>
          <div class="tl-org">{e(exp['org'])}</div>
          <ul class="tl-list">
{pts}
          </ul>
        </div>
      </div>"""
    return f"""
  <!-- EXPERIENCE -->
  <section id="experience" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">02 / experience</span>
        <h2>Work Experience</h2>
      </div>
      <div class="timeline">{items}
      </div>
    </div>
  </section>"""


def _projects(d: dict) -> str:
    cards = ""
    for proj in d["projects"]:
        img  = (f'<img src="{e(proj["img"])}" alt="{e(proj["title"])}" loading="lazy">'
                if proj.get("img") else '<div class="proj-img-ph">&#9881;</div>')
        cat  = (f'<div class="proj-cat">{e(proj["category"])}</div>'
                if proj.get("category") else "")
        desc = (f'<div class="proj-desc">{e(proj["description"])}</div>'
                if proj.get("description") else "")
        link = (f'<a href="{e(proj["github"])}" class="proj-link" target="_blank" rel="noopener">'
                f'<i class="fab fa-github"></i>&nbsp;GitHub</a>'
                if proj.get("github") else "")
        cards += f"""
      <div class="proj-card">
        <div class="proj-img">{img}<div class="proj-img-overlay"></div></div>
        <div class="proj-body">
          {cat}
          <div class="proj-title">{e(proj['title'])}</div>
          {desc}
          {link}
        </div>
      </div>"""
    return f"""
  <!-- PROJECTS -->
  <section id="projects" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">03 / projects</span>
        <h2>Featured Projects</h2>
      </div>
      <div class="projects-grid">{cards}
      </div>
    </div>
  </section>"""


def _skills(d: dict) -> str:
    blocks = ""
    for sk in d["skills"]:
        rows = "<br>".join(e(i) for i in sk["items"])
        blocks += f"""
      <div class="sk-block sk-{e(sk['color'])}">
        <div class="sk-title">{e(sk['title'])}</div>
        <div class="sk-items">{rows}</div>
      </div>"""
    return f"""
  <!-- SKILLS -->
  <section id="skills" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">04 / skills</span>
        <h2>Technical Skills</h2>
      </div>
      <div class="skills-grid">{blocks}
      </div>
    </div>
  </section>"""


def _education(d: dict) -> str:
    cards = "".join(f"""
      <div class="edu-card">
        <div class="edu-year">{e(edu['year'])}</div>
        <div class="edu-degree">{e(edu['degree'])}</div>
        <div class="edu-inst">{e(edu['institution'])}</div>
      </div>"""
        for edu in d["education"]
    )
    return f"""
  <!-- EDUCATION -->
  <section id="education" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">05 / education</span>
        <h2>Education</h2>
      </div>
      <div class="edu-grid">{cards}
      </div>
    </div>
  </section>"""


def _contact(d: dict) -> str:
    p = d["personal"]
    kaggle = (
        f'<a href="{e(p["kaggle"])}" class="c-link" target="_blank" rel="noopener">'
        f'<i class="fab fa-kaggle"></i> Kaggle</a>'
        if p.get("kaggle") else ""
    )
    return f"""
  <!-- CONTACT -->
  <section id="contact" class="section reveal">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">06 / contact</span>
        <h2>Get In Touch</h2>
      </div>
      <div class="contact-grid">
        <div class="contact-text">
          <h3>Let's work together.</h3>
          <p>Whether you're looking to automate CAE workflows, integrate AI into simulation
             pipelines, or collaborate on a research project — I'd love to hear from you.</p>
          <div class="contact-links">
            <a href="mailto:{e(p['email'])}" class="c-link">
              <i class="fas fa-envelope"></i> Email</a>
            <a href="https://linkedin.com/in/{e(p['linkedin'])}" class="c-link"
               target="_blank" rel="noopener">
              <i class="fab fa-linkedin"></i> LinkedIn</a>
            <a href="https://github.com/{e(p['github'])}" class="c-link"
               target="_blank" rel="noopener">
              <i class="fab fa-github"></i> GitHub</a>
            {kaggle}
          </div>
        </div>
        <div class="contact-card">
          <p class="contact-card__email">{e(p['email'])}</p>
          <p class="contact-card__note">
            Based in Munich, Germany &amp; Multan, Pakistan.<br>
            Open to remote collaboration worldwide.
          </p>
        </div>
      </div>
    </div>
  </section>"""


# ── Blog pages ─────────────────────────────────────────────────────

def _blog_listing(posts: list, p: dict) -> str:
    nav = _navbar(p, active='blog', prefix='', on_index=False)
    cards = ''
    for post in sorted(posts, key=lambda x: x['date'], reverse=True):
        tags = ''.join(
            f'<span class="blog-tag">{e(t)}</span>'
            for t in post.get('tags', [])
        )
        cards += f"""
        <a class="blog-card" href="blog/{e(post['slug'])}.html">
          <div class="blog-card__date">{e(post['date'])}</div>
          <div class="blog-card__title">{e(post['title'])}</div>
          <div class="blog-card__excerpt">{e(post.get('excerpt', '').strip())}</div>
          <div class="blog-card__tags">{tags}</div>
          <div class="blog-card__more">Read more &#8594;</div>
        </a>"""

    head = _head(
        f"Blog — {e(p['first_name'])} {e(p['last_name'])}",
        f"Technical blog by {e(p['first_name'])} {e(p['last_name'])}",
        css='assets/css/portfolio.css',
    )
    return f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>
{_bg()}

{nav}

  <main class="pf-main">
    <section class="blog-listing-hero">
      <div class="container">
        <span class="section-tag">writing</span>
        <h1>Blog</h1>
        <p class="blog-listing-sub">Thoughts on AI, deep learning, and engineering automation.</p>
      </div>
    </section>

    <section class="section" style="padding-top:0">
      <div class="container">
        <div class="blog-grid">{cards}
        </div>
      </div>
    </section>
  </main>

{_footer(p)}

  <script src="assets/js/portfolio.js" defer></script>
</body>
</html>"""


def _blog_post(post: dict, p: dict) -> str:
    nav     = _navbar(p, active='blog', prefix='../', on_index=False)
    content = md_to_html(post.get('content', ''))
    tags    = ''.join(
        f'<span class="blog-tag">{e(t)}</span>'
        for t in post.get('tags', [])
    )
    head = _head(
        f"{e(post['title'])} — {e(p['first_name'])} {e(p['last_name'])}",
        e(post.get('excerpt', '').strip()),
        css='../assets/css/portfolio.css',
    )
    return f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>
{_bg()}

{nav}

  <main class="pf-main">
    <article class="post-page">
      <div class="post-wrap">
        <a class="post-back" href="../blog.html">&#8592; Back to Blog</a>
        <header class="post-header">
          <div class="post-meta">
            <span class="post-date">{e(post['date'])}</span>
          </div>
          <h1 class="post-title">{e(post['title'])}</h1>
          <div class="post-tags">{tags}</div>
        </header>
        <div class="post-content">
{content}
        </div>
      </div>
    </article>
  </main>

{_footer(p)}

  <script src="../assets/js/portfolio.js" defer></script>
</body>
</html>"""


# ── Full page (index.html) ─────────────────────────────────────────

def _page(d: dict) -> str:
    p          = d["personal"]
    typed_json = json.dumps(d["hero"]["typed_lines"], ensure_ascii=False)
    nav        = _navbar(p, on_index=True)

    body = "\n".join([
        _hero(d), _about(d), _experience(d),
        _projects(d), _skills(d), _education(d), _contact(d),
    ])

    head = _head(
        f"{e(p['first_name'])} {e(p['last_name'])} — {e(p['title'])}",
        e(p['description']),
    )

    return f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>

{_bg()}

{nav}

  <main class="pf-main">
{body}
  </main>

{_footer(p)}

  <!-- Pass typed-text data to JS -->
  <script>window.__TYPED__ = {typed_json};</script>
  <script src="assets/js/portfolio.js" defer></script>

</body>
</html>"""


# ── Build & Serve ──────────────────────────────────────────────────

def build() -> None:
    print("🔨  Reading data/portfolio.yml …")
    if not DATA.exists():
        sys.exit(f"  ✗  {DATA} not found")
    with open(DATA, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print("📝  Generating index.html …")
    page = _page(data)
    OUT.write_text(page, encoding="utf-8")
    print(f"✅  index.html  ({len(page.encode()) // 1024} KB)")

    # Blog pages — prefer content/blogs/*.md, fall back to data/blogs.yml
    md_posts = load_markdown_posts()
    if md_posts:
        print(f"🔨  Loading {len(md_posts)} posts from content/blogs/ …")
        posts = md_posts
    elif BLOGS.exists():
        print("🔨  Reading data/blogs.yml …")
        with open(BLOGS, encoding="utf-8") as f:
            blog_data = yaml.safe_load(f)
        posts = blog_data.get("posts", [])
    else:
        print("ℹ️   No blog sources found — skipping blog generation")
        return

    print(f"📝  Generating blog.html ({len(posts)} posts) …")
    BLOG_OUT.write_text(_blog_listing(posts, data["personal"]), encoding="utf-8")

    BLOG_DIR.mkdir(exist_ok=True)
    for post in posts:
        slug = post["slug"]
        dest = BLOG_DIR / f"{slug}.html"
        dest.write_text(_blog_post(post, data["personal"]), encoding="utf-8")
        print(f"   → blog/{slug}.html")

    print(f"✅  Blog: blog.html + {len(posts)} post pages")
    print(f"   {len(data['experience'])} experience · "
          f"{len(data['projects'])} projects · "
          f"{len(data['skills'])} skill blocks · "
          f"{len(data['education'])} education")


def serve(port: int = 8000) -> None:
    os.chdir(ROOT)

    class Silent(SimpleHTTPRequestHandler):
        def log_message(self, *_): pass

    print(f"\n🌐  http://localhost:{port}   (Ctrl-C to stop)\n")
    try:
        HTTPServer(("", port), Silent).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    build()
    if "-s" in sys.argv or "--serve" in sys.argv:
        serve()
