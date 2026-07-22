(() => {
  const body = document.body;
  if (!body) {
    return;
  }

  const LAST_REVIEW_KEY = "plc:last-review";

  const getSessionStorage = () => {
    try {
      return window.sessionStorage;
    } catch (_error) {
      return null;
    }
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

  const setBanner = (node, text, kind) => {
    if (!(node instanceof HTMLElement)) {
      return;
    }
    node.textContent = text;
    node.className = `status-banner is-visible ${kind}`;
  };

  const initLastReviewNotice = () => {
    const notice = document.querySelector("[data-last-review-notice]");
    const message = notice ? notice.querySelector("[data-last-review-message]") : null;
    const link = notice ? notice.querySelector("[data-last-review-link]") : null;
    const storage = getSessionStorage();
    if (
      !(notice instanceof HTMLElement)
      || !(message instanceof HTMLElement)
      || !(link instanceof HTMLAnchorElement)
      || !storage
    ) {
      return;
    }
    const rawValue = storage.getItem(LAST_REVIEW_KEY);
    if (!rawValue) {
      return;
    }
    storage.removeItem(LAST_REVIEW_KEY);
    try {
      const saved = JSON.parse(rawValue);
      const correctionUrl = normalizeInternalUrl(saved.correctionUrl);
      const summary = String(saved.summary || "").trim();
      if (!correctionUrl || !summary) {
        return;
      }
      message.textContent = summary;
      link.href = correctionUrl;
      notice.hidden = false;
    } catch (_error) {
      return;
    }
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
    const yesButton = buttons.find((button) => String(button.dataset.decision || "") === "qualified_yes") || null;
    const rejectToggle = actionRoot.querySelector("[data-review-reject-toggle]");
    const rejectPicker = actionRoot.querySelector("[data-review-reject-picker]");
    const rejectCancel = actionRoot.querySelector("[data-review-reject-cancel]");
    const rejectSubmit = actionRoot.querySelector("[data-review-reject-submit]");
    const reviewClassInput = actionRoot.querySelector("[data-review-class-input]");
    const reviewClassPreview = actionRoot.querySelector("[data-review-class-preview]");
    const rejectReasonInputs = Array.from(actionRoot.querySelectorAll("[data-review-reject-reason]")).filter(
      (input) => input instanceof HTMLInputElement,
    );
    const modelClass = String(actionRoot.dataset.modelClass || "").trim();
    let rejectSubmissionPending = false;

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
        reviewClassPreview.textContent = `Jika diterima, data ini dihitung sebagai ${reviewedClass}. Jika ditolak, tipe ini tidak menjadi hasil operasional.`;
        return;
      }
      const detectedClass = modelClass || "detected class";
      reviewClassPreview.textContent = `Jika diterima tanpa koreksi, data ini dihitung sebagai ${detectedClass}. Jika ditolak, tidak ada class operasional baru yang dibuat.`;
    };

    const submitRejectReason = (input) => {
      if (
        rejectSubmissionPending
        || !(input instanceof HTMLInputElement)
        || !(rejectSubmit instanceof HTMLButtonElement)
      ) {
        return;
      }
      rejectSubmissionPending = true;
      input.checked = true;
      actionRoot.requestSubmit(rejectSubmit);
    };

    const setRejectPickerOpen = (isOpen, shouldFocus = false) => {
      if (!(rejectPicker instanceof HTMLFieldSetElement) || !(rejectToggle instanceof HTMLButtonElement)) {
        return;
      }
      rejectPicker.hidden = !isOpen;
      rejectToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
      rejectToggle.classList.toggle("is-active", isOpen);
      if (isOpen && shouldFocus) {
        const selected = rejectReasonInputs.find((input) => input.checked);
        const focusTarget = selected || rejectReasonInputs[0];
        if (focusTarget instanceof HTMLInputElement) {
          focusTarget.focus();
        }
      }
    };

    actionRoot.addEventListener("submit", (event) => {
      const submitter = event.submitter;
      if (!(submitter instanceof HTMLButtonElement) || !buttons.includes(submitter)) {
        return;
      }
      const storage = getSessionStorage();
      if (storage && currentEventUid) {
        const decision = String(submitter.dataset.decision || "");
        const selectedReason = rejectReasonInputs.find((input) => input.checked) || null;
        const reasonOption = selectedReason ? selectedReason.closest(".detail-reject-option") : null;
        const reasonLabelNode = reasonOption ? reasonOption.querySelector(".detail-reject-option-label") : null;
        const reasonLabel = reasonLabelNode ? String(reasonLabelNode.textContent || "").trim() : "";
        const correctionUrl = new URL(window.location.href);
        correctionUrl.searchParams.set("status", "all");
        const summary = decision === "qualified_no" && reasonLabel
          ? `Data terakhir ditolak: ${reasonLabel}.`
          : "Data terakhir diterima.";
        storage.setItem(LAST_REVIEW_KEY, JSON.stringify({
          correctionUrl: `${correctionUrl.pathname}${correctionUrl.search}`,
          summary,
        }));
      }
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

    if (rejectToggle instanceof HTMLButtonElement) {
      rejectToggle.addEventListener("click", () => {
        setRejectPickerOpen(true, true);
      });
    }

    if (rejectCancel instanceof HTMLButtonElement) {
      rejectCancel.addEventListener("click", () => {
        setRejectPickerOpen(false);
        if (rejectToggle instanceof HTMLButtonElement) {
          rejectToggle.focus();
        }
      });
    }

    if (rejectReasonInputs.length > 0) {
      rejectReasonInputs.forEach((input) => {
        input.addEventListener("change", () => {
          submitRejectReason(input);
        });
      });
    }

    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      if (event.key === "Escape" && rejectPicker instanceof HTMLFieldSetElement && !rejectPicker.hidden) {
        event.preventDefault();
        setRejectPickerOpen(false);
        if (rejectToggle instanceof HTMLButtonElement) {
          rejectToggle.focus();
        }
        return;
      }
      if (
        rejectPicker instanceof HTMLFieldSetElement
        && !rejectPicker.hidden
        && /^[1-6]$/.test(event.key)
      ) {
        const reasonIndex = Number.parseInt(event.key, 10) - 1;
        const reasonInput = rejectReasonInputs[reasonIndex];
        if (reasonInput instanceof HTMLInputElement) {
          event.preventDefault();
          submitRejectReason(reasonInput);
        }
        return;
      }
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
      } else if (key === "n" && rejectToggle instanceof HTMLButtonElement) {
        event.preventDefault();
        setRejectPickerOpen(true, true);
      } else if (key === "j" && nextDetailUrl && isDetailPage) {
        event.preventDefault();
        window.location.assign(nextDetailUrl);
      } else if (key === "k" && previousDetailUrl && isDetailPage) {
        event.preventDefault();
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
  initLastReviewNotice();
  initReviewActions();
  initReviewQueueSelection();
})();
