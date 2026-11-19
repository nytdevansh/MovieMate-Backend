/* Response-time accessibility and visual polish for the preserved Frontend pages.
 *
 * The original HTML/CSS files stay byte-for-byte unchanged. This script adds a
 * skip link, visible focus rings, reduced-motion support, and fixes a few small
 * target-size and labelling issues found in the Apple HIG audit.
 */
(function () {
  "use strict";

  var style = document.createElement("style");
  style.id = "mm-ui-polish-style";
  style.textContent = [
    ":root{color-scheme:dark}",
    ":where(a,button,input,select,textarea,[role=button],[tabindex]):focus-visible{outline:2px solid #ff8a65;outline-offset:3px}",
    "@media (prefers-reduced-motion: reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}}",
    ".mm-skip-link{position:fixed;left:12px;top:-64px;z-index:20000;padding:10px 14px;border-radius:10px;background:#fff;color:#111;text-decoration:none;font:600 14px/1.2 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)}",
    ".mm-skip-link:focus{top:12px}",
    "body.mm-signin header{display:flex;align-items:center;padding:14px 20px;border-bottom-color:#2a2a2a}",
    "body.mm-signin main{width:min(400px,calc(100vw - 32px));max-width:none;margin:clamp(40px,10vh,100px) auto 0;padding:0 0 48px}",
    "body.mm-signin h1{font-size:clamp(1.6rem,4vw,2rem);line-height:1.2;letter-spacing:-.02em}",
    "body.mm-signin p{line-height:1.5}",
    "body.mm-signin form{display:grid;gap:12px}",
    "body.mm-signin input,body.mm-signin button{width:100%;max-width:100%;min-height:56px;border-radius:12px;font:inherit}",
    "body.mm-signin input{border:1px solid #404040}",
    "body.mm-signin button{color:#fff;cursor:pointer;border:0}",
    "body.mm-signin button:hover{filter:brightness(1.08)}",
    "body.mm-signin main>a{display:inline-flex;align-items:center;min-height:44px;margin-top:8px}",
    "body.mm-admin [class*='text-[10px]']{font-size:11px!important}",
    "body.mm-admin button:focus-visible,body.mm-admin a:focus-visible{outline-color:#ffc085}"
  ].join("\n");
  document.head.appendChild(style);

  function addSkipLink() {
    if (document.querySelector(".mm-skip-link")) return;
    var target = document.querySelector("main, .main-content, [role='main']");
    if (!target) return;
    if (!target.id) target.id = "mm-main-content";
    var link = document.createElement("a");
    link.className = "mm-skip-link";
    link.href = "#" + target.id;
    link.textContent = "Skip to main content";
    target.parentNode.insertBefore(link, target);
  }

  function polishSignIn() {
    if (document.title !== "Sign in") return;
    var input = document.querySelector("main input");
    if (!input) return;
    document.body.classList.add("mm-signin");
    input.setAttribute("type", "email");
    input.setAttribute("aria-label", "Email address");
    input.setAttribute("autocomplete", "email");
    input.setAttribute("placeholder", "name@example.com");
    input.setAttribute("required", "");
    var button = document.querySelector("main button[type='submit']");
    if (button) button.setAttribute("aria-label", "Continue to the dashboard");
  }

  function polishAdmin() {
    var toggle = document.querySelector('[onclick="toggleMobileSidebar()"]');
    if (!toggle) return;
    document.body.classList.add("mm-admin");
    if (!toggle.textContent.trim() && !toggle.getAttribute("aria-label")) {
      toggle.setAttribute("aria-label", "Open navigation menu");
    }
    toggle.addEventListener("click", function () {
      var open = document.querySelector(".sidebar") &&
        document.querySelector(".sidebar").classList.contains("translate-x-0");
      toggle.setAttribute("aria-expanded", open ? "false" : "true");
    });
  }

  addSkipLink();
  polishSignIn();
  polishAdmin();
})();
