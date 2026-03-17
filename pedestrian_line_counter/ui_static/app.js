(() => {
  const body = document.body;
  if (!body) {
    return;
  }

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
    if (!(actionRoot instanceof HTMLElement)) {
      return;
    }
    const buttons = Array.from(actionRoot.querySelectorAll("[data-review-submit]")).filter(
      (button) => button instanceof HTMLButtonElement,
    );
    if (buttons.length === 0) {
      return;
    }
    const feedback = document.querySelector("[data-review-feedback]");
    const notes = document.querySelector("[data-review-notes]");
    const currentEventUid = actionRoot.dataset.currentEvent || "";
    const nextEventUid = actionRoot.dataset.nextEvent || "";
    const cameraId = actionRoot.dataset.cameraId || "";
    const statusFilter = actionRoot.dataset.statusFilter || "pending";
    const redirectBase = actionRoot.dataset.reviewBase || "/ui/review";

    const submitReview = async (decision, reloadMode) => {
      if (!currentEventUid) {
        return;
      }
      const notesText = notes instanceof HTMLTextAreaElement ? notes.value : "";
      buttons.forEach((button) => {
        button.disabled = true;
      });

      try {
        const response = await fetch(`/events/${currentEventUid}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decision, notes: notesText }),
        });
        if (!response.ok) {
          throw new Error("review update failed");
        }
        if (reloadMode === "reload") {
          window.location.reload();
          return;
        }
        const payload = await response.json();
        const targetEvent = nextEventUid || payload.next_event_uid;
        const params = new URLSearchParams();
        if (cameraId) params.set("camera_id", cameraId);
        if (statusFilter) params.set("status", statusFilter);
        if (targetEvent) params.set("event_uid", targetEvent);
        const nextUrl = params.toString() ? `${redirectBase}?${params.toString()}` : redirectBase;
        window.location.assign(nextUrl);
      } catch (_error) {
        setBanner(feedback, "Review update failed.", "error");
        buttons.forEach((button) => {
          button.disabled = false;
        });
      }
    };

    actionRoot.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const button = target.closest("[data-review-submit]");
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      submitReview(String(button.dataset.decision || ""), button.dataset.reloadMode || "");
    });

    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      if (active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "y") {
        event.preventDefault();
        submitReview("qualified_yes", body.classList.contains("page-event-detail") ? "reload" : "");
      } else if (key === "n") {
        event.preventDefault();
        submitReview("qualified_no", body.classList.contains("page-event-detail") ? "reload" : "");
      } else if (key === "j" && nextEventUid && body.classList.contains("page-review")) {
        event.preventDefault();
        const params = new URLSearchParams();
        if (cameraId) params.set("camera_id", cameraId);
        if (statusFilter) params.set("status", statusFilter);
        params.set("event_uid", nextEventUid);
        window.location.assign(`${redirectBase}?${params.toString()}`);
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
          const positionText = `${index + 1} / ${rows.length}`;
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
  initReviewActions();
  initReviewQueueSelection();
})();
