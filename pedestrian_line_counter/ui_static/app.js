(() => {
  const body = document.body;
  if (!body) {
    return;
  }

  const DETAIL_SCROLL_Y_KEY = "plc:event-detail:scroll-y";
  const DETAIL_SCROLL_PENDING_KEY = "plc:event-detail:scroll-pending";

  const getSessionStorage = () => {
    try {
      return window.sessionStorage;
    } catch (_error) {
      return null;
    }
  };

  const clearDetailScrollState = () => {
    const storage = getSessionStorage();
    if (!storage) {
      return;
    }
    storage.removeItem(DETAIL_SCROLL_Y_KEY);
    storage.removeItem(DETAIL_SCROLL_PENDING_KEY);
  };

  const saveDetailScrollState = () => {
    const storage = getSessionStorage();
    if (!storage || !body.classList.contains("page-event-detail")) {
      return;
    }
    storage.setItem(DETAIL_SCROLL_Y_KEY, String(Math.max(0, Math.round(window.scrollY || window.pageYOffset || 0))));
    storage.setItem(DETAIL_SCROLL_PENDING_KEY, "1");
  };

  const normalizeInternalUrl = (value) => {
    if (!value) {
      return "";
    }
    try {
      const parsed = new URL(value, window.location.href);
      if (parsed.origin !== window.location.origin) {
        return "";
      }
      return `${parsed.pathname}${parsed.search}${parsed.hash}`;
    } catch (_error) {
      return "";
    }
  };

  const restoreDetailScrollState = () => {
    if (!body.classList.contains("page-event-detail")) {
      clearDetailScrollState();
      return;
    }
    const storage = getSessionStorage();
    if (!storage || storage.getItem(DETAIL_SCROLL_PENDING_KEY) !== "1") {
      return;
    }
    const rawValue = storage.getItem(DETAIL_SCROLL_Y_KEY) || "";
    const targetScrollY = Number.parseInt(rawValue, 10);
    if (!Number.isFinite(targetScrollY) || targetScrollY < 1) {
      clearDetailScrollState();
      return;
    }

    let attempts = 0;
    let settled = false;

    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      clearDetailScrollState();
    };

    const attemptRestore = () => {
      if (settled) {
        return;
      }
      attempts += 1;
      window.scrollTo(0, targetScrollY);

      const currentY = Math.round(window.scrollY || window.pageYOffset || 0);
      const maxScrollY = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      if (Math.abs(currentY - targetScrollY) <= 2) {
        finish();
        return;
      }
      if (attempts >= 12 && maxScrollY >= targetScrollY - 2) {
        finish();
        return;
      }
      if (attempts >= 20) {
        finish();
        return;
      }
      window.setTimeout(attemptRestore, 60);
    };

    const evidenceImage = document.querySelector(".detail-player .evidence-image");
    if (evidenceImage instanceof HTMLImageElement && !evidenceImage.complete) {
      evidenceImage.addEventListener("load", attemptRestore, { once: true });
      evidenceImage.addEventListener("error", attemptRestore, { once: true });
    }
    window.addEventListener("load", attemptRestore, { once: true });
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(attemptRestore);
    });
  };

  const setBanner = (node, text, kind) => {
    if (!(node instanceof HTMLElement)) {
      return;
    }
    node.textContent = text;
    node.className = `status-banner is-visible ${kind}`;
  };

  const setLoginStatus = (node, text, kind) => {
    if (!(node instanceof HTMLElement)) {
      return;
    }
    const paragraph = node.querySelector("p");
    if (paragraph instanceof HTMLParagraphElement) {
      paragraph.textContent = text;
    } else {
      node.textContent = text;
    }
    node.classList.remove("error", "ok");
    if (kind) {
      node.classList.add(kind);
    }
  };

  const initLogout = () => {
    const link = document.getElementById("ui-logout-link");
    if (!(link instanceof HTMLAnchorElement)) {
      return;
    }
    link.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await fetch("/api/auth/logout", { method: "POST" });
      } finally {
        window.location.assign(link.href);
      }
    });
  };

  const initLogin = () => {
    if (!body.classList.contains("page-login")) {
      return;
    }
    const loginForm = document.getElementById("login-form");
    const statusBox = document.getElementById("login-status");
    if (!(loginForm instanceof HTMLFormElement) || !(statusBox instanceof HTMLElement)) {
      return;
    }
    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");
    const submitBtn = loginForm.querySelector("button[type='submit']");
    const nextPath = loginForm.dataset.nextPath || "/ui/dashboard";

    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const username = usernameInput instanceof HTMLInputElement ? String(usernameInput.value || "").trim() : "";
      const password = passwordInput instanceof HTMLInputElement ? String(passwordInput.value || "") : "";
      if (!username || !password) {
        setLoginStatus(statusBox, "Enter username and password.", "error");
        return;
      }
      if (submitBtn instanceof HTMLButtonElement) {
        submitBtn.disabled = true;
      }
      setLoginStatus(statusBox, "Signing in...", "ok");
      try {
        const response = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        if (!response.ok) {
          throw new Error("login failed");
        }
        window.location.assign(nextPath);
      } catch (_error) {
        setLoginStatus(statusBox, "Login failed. Check your credentials.", "error");
        if (submitBtn instanceof HTMLButtonElement) {
          submitBtn.disabled = false;
        }
      }
    });
  };

  const initReviewActions = () => {
    const actionRoot = document.querySelector("[data-review-actions]");
    if (!(actionRoot instanceof HTMLFormElement)) {
      return;
    }
    const buttons = Array.from(actionRoot.querySelectorAll("[data-review-submit]")).filter(
      (button) => button instanceof HTMLButtonElement,
    );
    if (buttons.length === 0) {
      return;
    }
    const feedback = document.querySelector("[data-review-feedback]");
    const currentEventUid = actionRoot.dataset.currentEvent || "";
    const previousDetailUrl = actionRoot.dataset.previousDetailUrl || "";
    const nextDetailUrl = actionRoot.dataset.nextDetailUrl || "";
    const isDetailPage = body.classList.contains("page-event-detail");
    const detailNavigationTargets = new Set(
      [previousDetailUrl, nextDetailUrl]
        .map((value) => normalizeInternalUrl(value))
        .filter(Boolean),
    );
    const yesButton = buttons.find((button) => String(button.dataset.decision || "") === "qualified_yes") || null;
    const noButton = buttons.find((button) => String(button.dataset.decision || "") === "qualified_no") || null;
    const reviewClassInput = actionRoot.querySelector("[data-review-class-input]");
    const reviewClassPreview = actionRoot.querySelector("[data-review-class-preview]");
    const modelClass = String(actionRoot.dataset.modelClass || "").trim();

    const updateReviewClassPreview = () => {
      if (
        !(
          reviewClassInput instanceof HTMLInputElement
          || reviewClassInput instanceof HTMLSelectElement
        )
        || !(reviewClassPreview instanceof HTMLElement)
      ) {
        return;
      }
      const reviewedClass = String(reviewClassInput.value || "").trim();
      if (reviewedClass) {
        reviewClassPreview.textContent = `Jika disimpan YES, queue/API akan memakai class operasional ${reviewedClass}. Jika disimpan NO, correction ini tetap disimpan sebagai audit.`;
        return;
      }
      const detectedClass = modelClass || "detected class";
      reviewClassPreview.textContent = `Jika field correction dikosongkan lalu disimpan YES, queue/API tetap memakai detected class ${detectedClass}. Jika disimpan NO, tidak ada override class operasional yang dipakai.`;
    };

    actionRoot.addEventListener("submit", (event) => {
      const submitter = event.submitter;
      if (!(submitter instanceof HTMLButtonElement) || !buttons.includes(submitter)) {
        return;
      }
      saveDetailScrollState();
      setBanner(feedback, "Saving review…", "info");
      window.setTimeout(() => {
        buttons.forEach((button) => {
          button.disabled = true;
        });
      }, 0);
    });

    if (reviewClassInput instanceof HTMLInputElement || reviewClassInput instanceof HTMLSelectElement) {
      reviewClassInput.addEventListener("input", updateReviewClassPreview);
      reviewClassInput.addEventListener("change", updateReviewClassPreview);
      updateReviewClassPreview();
    }

    if (isDetailPage && detailNavigationTargets.size > 0) {
      document.addEventListener("click", (event) => {
        if (event.defaultPrevented || event.button !== 0) {
          return;
        }
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return;
        }
        const target = event.target;
        if (!(target instanceof Element)) {
          return;
        }
        const link = target.closest("a[href]");
        if (!(link instanceof HTMLAnchorElement)) {
          return;
        }
        if (link.target && link.target !== "_self") {
          return;
        }
        const normalizedHref = normalizeInternalUrl(link.getAttribute("href") || "");
        if (!normalizedHref || !detailNavigationTargets.has(normalizedHref)) {
          return;
        }
        saveDetailScrollState();
      });
    }

    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      if (
        active instanceof HTMLTextAreaElement
        || active instanceof HTMLInputElement
        || active instanceof HTMLSelectElement
        || (active instanceof HTMLElement && active.isContentEditable)
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "y" && yesButton) {
        event.preventDefault();
        actionRoot.requestSubmit(yesButton);
      } else if (key === "n" && noButton) {
        event.preventDefault();
        actionRoot.requestSubmit(noButton);
      } else if (key === "j" && nextDetailUrl && isDetailPage) {
        event.preventDefault();
        saveDetailScrollState();
        window.location.assign(nextDetailUrl);
      } else if (key === "k" && previousDetailUrl && isDetailPage) {
        event.preventDefault();
        saveDetailScrollState();
        window.location.assign(previousDetailUrl);
      } else if (key === "enter" && currentEventUid && body.classList.contains("page-review")) {
        event.preventDefault();
        window.location.assign(`/ui/events/${encodeURIComponent(currentEventUid)}`);
      }
    });
  };

  const initReviewQueueSelection = () => {
    if (!body.classList.contains("page-review")) {
      return;
    }
    const browser = document.querySelector("[data-queue-browser]");
    if (!(browser instanceof HTMLElement)) {
      return;
    }
    const rows = Array.from(browser.querySelectorAll("[data-queue-row]")).filter(
      (row) => row instanceof HTMLElement,
    );
    if (rows.length === 0) {
      return;
    }

    const selectionInput = document.querySelector("[data-queue-selection-input]");
    const positionPill = document.querySelector("[data-queue-position-pill]");
    const positionValue = document.querySelector("[data-queue-position-value]");
    const footerSelection = document.querySelector("[data-queue-footer-selection]");
    const selectedLabel = document.querySelector("[data-queue-selected-label]");
    const selectedDetailLinks = Array.from(
      document.querySelectorAll("[data-queue-selected-detail], [data-queue-footer-detail]"),
    ).filter((node) => node instanceof HTMLAnchorElement);

    const updateSelectionUi = (row) => {
      if (!(row instanceof HTMLElement)) {
        return;
      }
      rows.forEach((candidate, index) => {
        const active = candidate === row;
        candidate.classList.toggle("queue-row-active", active);
        candidate.setAttribute("aria-current", active ? "true" : "false");
        const sessionLink = candidate.querySelector(".queue-session-link");
        if (sessionLink instanceof HTMLElement) {
          sessionLink.classList.toggle("active", active);
        }
        if (active) {
          const absoluteIndex = Number.parseInt(candidate.dataset.absoluteIndex || "", 10);
          const queueTotal = Number.parseInt(candidate.dataset.queueTotal || "", 10);
          const positionIndex = Number.isFinite(absoluteIndex) && absoluteIndex > 0 ? absoluteIndex : index + 1;
          const positionTotal = Number.isFinite(queueTotal) && queueTotal > 0 ? queueTotal : rows.length;
          const positionText = `${positionIndex} / ${positionTotal}`;
          if (selectionInput instanceof HTMLInputElement) {
            selectionInput.value = positionText;
          }
          if (positionPill instanceof HTMLElement) {
            positionPill.textContent = `item ${positionText}`;
          }
          if (positionValue instanceof HTMLElement) {
            positionValue.textContent = positionText;
          }
          if (footerSelection instanceof HTMLElement) {
            footerSelection.textContent = `Selected ${positionText}`;
          }
        }
      });

      const shortEvent = row.dataset.eventShort || row.dataset.eventUid || "No selection";
      if (selectedLabel instanceof HTMLElement) {
        selectedLabel.textContent = `Selected: ${shortEvent}`;
      }

      const detailUrl = row.dataset.detailUrl || "";
      selectedDetailLinks.forEach((link) => {
        link.href = detailUrl || "#";
        link.setAttribute("aria-disabled", detailUrl ? "false" : "true");
        link.classList.toggle("is-disabled", !detailUrl);
      });
    };

    const selectRow = (row, options = {}) => {
      if (!(row instanceof HTMLElement)) {
        return;
      }
      updateSelectionUi(row);
      if (options.focus) {
        row.focus();
      }
      const targetUrl = row.dataset.selectUrl || "";
      if (!options.skipHistory && targetUrl && window.history && typeof window.history.replaceState === "function") {
        window.history.replaceState({}, "", targetUrl);
      }
    };

    const openRowDetail = (row) => {
      if (!(row instanceof HTMLElement)) {
        return;
      }
      const detailUrl = row.dataset.detailUrl || "";
      if (detailUrl) {
        window.location.assign(detailUrl);
      }
    };

    browser.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const actionLink = target.closest(".btn-compact");
      if (actionLink instanceof HTMLAnchorElement) {
        return;
      }
      const row = target.closest("[data-queue-row]");
      if (!(row instanceof HTMLElement)) {
        return;
      }
      event.preventDefault();
      selectRow(row, { focus: false });
    });

    browser.addEventListener("keydown", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (!target.matches("[data-queue-row]")) {
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        selectRow(target, { focus: true });
      } else if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        openRowDetail(target);
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.defaultPrevented) {
        return;
      }
      const active = document.activeElement;
      if (
        active instanceof HTMLTextAreaElement
        || active instanceof HTMLInputElement
        || active instanceof HTMLSelectElement
        || active instanceof HTMLButtonElement
        || active instanceof HTMLAnchorElement
      ) {
        return;
      }
      const currentIndex = rows.findIndex((row) => row.classList.contains("queue-row-active"));
      if (currentIndex < 0) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "j" && currentIndex + 1 < rows.length) {
        event.preventDefault();
        selectRow(rows[currentIndex + 1], { focus: true });
      } else if (key === "k" && currentIndex > 0) {
        event.preventDefault();
        selectRow(rows[currentIndex - 1], { focus: true });
      } else if (key === "enter") {
        event.preventDefault();
        openRowDetail(rows[currentIndex]);
      }
    });

    const activeRow = rows.find((row) => row.classList.contains("queue-row-active")) || rows[0];
    if (activeRow) {
      selectRow(activeRow, { skipHistory: true });
    }
  };

  initLogout();
  initLogin();
  restoreDetailScrollState();
  initReviewActions();
  initReviewQueueSelection();
})();
