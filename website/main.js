(function () {
  "use strict";

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  const QUERY_TOKENS_BEFORE = 1597;
  const QUERY_TOKENS_AFTER = 416;
  const SAVINGS_PCT = Math.round(
    ((QUERY_TOKENS_BEFORE - QUERY_TOKENS_AFTER) / QUERY_TOKENS_BEFORE) * 100
  );

  /* Mobile nav */
  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.getElementById("site-nav");

  if (navToggle && siteNav) {
    navToggle.addEventListener("click", function () {
      const open = siteNav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });

    siteNav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        siteNav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* Compare — clip-path reveal + shadcn-style slider (rAF-smoothed) */
  const compareRoot = document.querySelector("[data-compare]");
  const compareSlider = document.getElementById("compare-slider");
  const compareViewport = document.querySelector("[data-compare-viewport]");
  const compareKnob = document.querySelector(".compare-divider-knob");
  const metricBefore = document.getElementById("metric-before");
  const metricAfter = document.getElementById("metric-after");
  const savingsPct = document.getElementById("savings-pct");

  let comparePct = 48;
  let compareTarget = 48;
  let compareRaf = 0;
  let compareDragging = false;

  function clampCompare(pct) {
    return Math.min(100, Math.max(0, pct));
  }

  function applyCompareImmediate(pct) {
    const clamped = clampCompare(pct);
    comparePct = clamped;
    compareTarget = clamped;
    const pos = clamped + "%";

    if (compareRoot) {
      compareRoot.style.setProperty("--compare-pos", pos);
    }
    if (compareSlider) {
      compareSlider.value = String(Math.round(clamped));
      compareSlider.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }
    if (compareKnob) {
      compareKnob.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }

    if (metricBefore && metricAfter) {
      const t = clamped / 100;
      const liveBefore = Math.round(
        QUERY_TOKENS_BEFORE * (1 - t * 0.35) + QUERY_TOKENS_AFTER * t * 0.35
      );
      const liveAfter = Math.round(
        QUERY_TOKENS_AFTER + (QUERY_TOKENS_BEFORE - QUERY_TOKENS_AFTER) * (1 - t) * 0.12
      );
      metricBefore.textContent = liveBefore.toLocaleString();
      metricAfter.textContent = liveAfter.toLocaleString();
    }
  }

  function setCompareTarget(pct, immediate) {
    compareTarget = clampCompare(pct);
    if (immediate || prefersReducedMotion) {
      if (compareRaf) {
        cancelAnimationFrame(compareRaf);
        compareRaf = 0;
      }
      applyCompareImmediate(compareTarget);
      return;
    }
    if (!compareRaf) {
      compareRaf = requestAnimationFrame(tickCompare);
    }
  }

  function tickCompare() {
    const delta = compareTarget - comparePct;
    if (Math.abs(delta) < 0.35) {
      applyCompareImmediate(compareTarget);
      compareRaf = 0;
      return;
    }
    applyCompareImmediate(comparePct + delta * 0.22);
    compareRaf = requestAnimationFrame(tickCompare);
  }

  function setDragging(active) {
    compareDragging = active;
    if (compareRoot) {
      compareRoot.classList.toggle("is-dragging", active);
    }
  }

  function positionFromClientX(clientX) {
    if (!compareViewport) return;
    const rect = compareViewport.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    setCompareTarget(ratio * 100, true);
  }

  if (compareSlider && compareRoot) {
    compareSlider.addEventListener("input", function () {
      setCompareTarget(Number(compareSlider.value), false);
    });
    applyCompareImmediate(Number(compareSlider.value) || 48);
  }

  if (savingsPct) {
    savingsPct.textContent = String(SAVINGS_PCT);
  }

  if (compareViewport) {
    function onPointerDown(e) {
      if (e.target.closest(".compare-footer")) return;
      setDragging(true);
      compareViewport.setPointerCapture(e.pointerId);
      positionFromClientX(e.clientX);
    }

    function onPointerMove(e) {
      if (!compareDragging) return;
      positionFromClientX(e.clientX);
    }

    function onPointerUp(e) {
      setDragging(false);
      try {
        compareViewport.releasePointerCapture(e.pointerId);
      } catch (_) {
        /* ignore */
      }
    }

    compareViewport.addEventListener("pointerdown", onPointerDown);
    compareViewport.addEventListener("pointermove", onPointerMove);
    compareViewport.addEventListener("pointerup", onPointerUp);
    compareViewport.addEventListener("pointercancel", onPointerUp);
  }

  if (compareKnob) {
    compareKnob.addEventListener("keydown", function (e) {
      let next = compareTarget;
      if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        next -= 4;
      } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        next += 4;
      } else if (e.key === "Home") {
        next = 0;
      } else if (e.key === "End") {
        next = 100;
      } else {
        return;
      }
      e.preventDefault();
      setCompareTarget(next, false);
    });
  }

  /* Scroll reveal */
  const revealEls = document.querySelectorAll(".reveal");

  if (revealEls.length && !prefersReducedMotion) {
    const io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -6% 0px", threshold: 0.08 }
    );
    revealEls.forEach(function (el) {
      io.observe(el);
    });
  } else {
    revealEls.forEach(function (el) {
      el.classList.add("is-visible");
    });
  }

  /* Metric counters — progressive enhancement only */
  const counters = document.querySelectorAll("[data-count-to]");

  function animateCounter(el) {
    const target = parseFloat(el.getAttribute("data-count-to"), 10);
    const suffix = el.getAttribute("data-suffix") || "";
    const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
    const duration = 900;
    const start = performance.now();

    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const val = target * eased;
      el.textContent = val.toFixed(decimals) + suffix;
      if (t < 1) {
        requestAnimationFrame(tick);
      }
    }

    if (prefersReducedMotion || Number.isNaN(target)) {
      return;
    }

    el.textContent = (0).toFixed(decimals) + suffix;
    requestAnimationFrame(tick);
  }

  if (counters.length && !prefersReducedMotion) {
    const counterObs = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            counterObs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach(function (c) {
      counterObs.observe(c);
    });
  }

  /* Header shadow on scroll */
  const header = document.querySelector(".site-header");
  if (header) {
    function onScroll() {
      header.classList.toggle("is-scrolled", window.scrollY > 8);
    }
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }
})();
