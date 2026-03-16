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

  initLogout();
  initLogin();
  initReviewActions();
})();
