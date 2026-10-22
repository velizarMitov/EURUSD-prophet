// The app shell's behaviour (openspec change app-shell-and-gold-history).
// Injected into every page route by api.py's ShellLayer; served from /ui/shell/.
//
//   language  BG/EN, remembered as 'prophet.lang'. The home page's older key
//             'eurusd.home.lang' is honoured once, so an earlier choice survives.
//             A switch updates the bar's labels and <html lang>, then fires
//             'ps:lang' (detail: 'bg' | 'en') for pages with their own dictionary.
//   theme     System / Light / Dark, remembered as 'prophet.theme'. The head
//             snippet applies it before first paint; this file wires the control.
//   menu      below 860 px the links fold behind a menu button; Escape closes
//             it and returns focus to the button that opened it.
//
// Storage can throw (private mode, blocked site data): every access is guarded,
// and the fallback is Bulgarian and the system theme.
(function () {
  'use strict';
  var LANG_KEY = 'prophet.lang', OLD_LANG_KEY = 'eurusd.home.lang', THEME_KEY = 'prophet.theme';
  var root = document.documentElement;

  function read(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function write(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* this visit only */ }
  }

  function storedLang() {
    var v = read(LANG_KEY);
    if (v !== 'bg' && v !== 'en') v = read(OLD_LANG_KEY);
    return v === 'en' ? 'en' : 'bg';
  }

  var lang = storedLang();

  function applyLang(next, announce) {
    lang = next === 'en' ? 'en' : 'bg';
    root.setAttribute('lang', lang);
    var shell = document.querySelector('[data-ps-shell]');
    if (shell) {
      shell.querySelectorAll('[data-bg][data-en]').forEach(function (el) {
        el.textContent = el.getAttribute('data-' + lang);
      });
      shell.querySelectorAll('[data-ps-lang]').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-ps-lang') === lang));
      });
    }
    if (announce) document.dispatchEvent(new CustomEvent('ps:lang', { detail: lang }));
  }

  function applyTheme(choice) {
    if (choice === 'light' || choice === 'dark') root.setAttribute('data-theme', choice);
    else { choice = 'system'; root.removeAttribute('data-theme'); }
    document.querySelectorAll('[data-ps-theme]').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-ps-theme') === choice));
    });
  }

  function disclosure(button, panel, onOpen) {
    function set(open) {
      button.setAttribute('aria-expanded', String(open));
      if (panel.hasAttribute('hidden') || panel.tagName === 'UL') panel.hidden = !open;
      panel.classList.toggle('ps-open', open);
      if (open && onOpen) onOpen();
    }
    button.addEventListener('click', function () { set(button.getAttribute('aria-expanded') !== 'true'); });
    panel.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { set(false); button.focus(); }
    });
    button.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') set(false);
    });
    document.addEventListener('click', function (e) {
      if (!button.contains(e.target) && !panel.contains(e.target)) set(false);
    });
    return set;
  }

  // Pages read this before their own first render, so they open in the
  // shell's language even though this script loads deferred.
  window.PS_SHELL = {
    lang: function () { return lang; },
    setLang: function (next) { write(LANG_KEY, next); applyLang(next, true); }
  };

  function init() {
    var shell = document.querySelector('[data-ps-shell]');
    if (!shell) return;
    applyLang(lang, true);
    var t = read(THEME_KEY);
    applyTheme(t);

    shell.querySelectorAll('[data-ps-lang]').forEach(function (b) {
      b.addEventListener('click', function () { window.PS_SHELL.setLang(b.getAttribute('data-ps-lang')); });
    });
    shell.querySelectorAll('[data-ps-theme]').forEach(function (b) {
      b.addEventListener('click', function () {
        var choice = b.getAttribute('data-ps-theme');
        write(THEME_KEY, choice);
        applyTheme(choice);
      });
    });

    var menuBtn = shell.querySelector('.ps-menu-btn'), menu = shell.querySelector('.ps-menu');
    if (menuBtn && menu) disclosure(menuBtn, menu, function () {
      var first = menu.querySelector('a,button');
      if (first) first.focus();
    });
    var moreBtn = shell.querySelector('.ps-more-btn'), more = shell.querySelector('.ps-more-list');
    if (moreBtn && more) disclosure(moreBtn, more, function () {
      var first = more.querySelector('a');
      if (first) first.focus();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
