/* ================================================================
   Portfolio JS — Typing · Scroll Reveal · Active Nav · Menu
   ================================================================ */

'use strict';

// ── Typing animation ──────────────────────────────────────────────
(function typed() {
  const el = document.querySelector('.typed-text');
  if (!el) return;

  const lines = window.__TYPED__ || ['AI & Automation Engineer'];
  let li = 0, ci = 0, del = false;

  function tick() {
    const str = lines[li];
    el.textContent = del ? str.slice(0, --ci) : str.slice(0, ++ci);

    let wait = del ? 38 : 68;
    if (!del && ci >= str.length)  { wait = 2400; del = true; }
    else if (del && ci <= 0)       { del = false; li = (li + 1) % lines.length; wait = 420; }
    setTimeout(tick, wait);
  }

  setTimeout(tick, 900);
})();

// ── Scroll reveal ─────────────────────────────────────────────────
// Runs AFTER stagger() has added .reveal to cards (see bottom of file)
function initReveal() {
  const items = document.querySelectorAll('.reveal');
  if (!items.length) return;

  const io = new IntersectionObserver(
    (entries) => entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        io.unobserve(entry.target);
      }
    }),
    { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
  );

  items.forEach(el => io.observe(el));
}

// ── Active nav on scroll ──────────────────────────────────────────
(function activeNav() {
  const sections = document.querySelectorAll('section[id]');
  const links    = document.querySelectorAll('.pf-nav__links a');
  if (!sections.length || !links.length) return;

  const io = new IntersectionObserver(
    (entries) => entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const a = document.querySelector(`.pf-nav__links a[href="#${e.target.id}"]`);
        if (a) a.classList.add('active');
      }
    }),
    { threshold: 0.35 }
  );

  sections.forEach(s => io.observe(s));
})();

// ── Mobile menu ───────────────────────────────────────────────────
(function mobileMenu() {
  const btn   = document.querySelector('.pf-nav__burger');
  const links = document.querySelector('.pf-nav__links');
  if (!btn || !links) return;

  btn.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    btn.setAttribute('aria-expanded', open);
  });

  links.querySelectorAll('a').forEach(a =>
    a.addEventListener('click', () => links.classList.remove('open'))
  );

  document.addEventListener('click', e => {
    if (!btn.contains(e.target) && !links.contains(e.target))
      links.classList.remove('open');
  });
})();

// ── Stagger: add .reveal + delay to cards, THEN init observer ─────
(function stagger() {
  const cards = document.querySelectorAll('.proj-card, .sk-block, .tl-item, .edu-card');
  cards.forEach((el, i) => {
    el.classList.add('reveal');
    el.style.transitionDelay = (i % 5) * 0.08 + 's';
  });

  // Now that .reveal is on all elements, start observing everything
  initReveal();
})();
