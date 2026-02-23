(function () {
  const lockFormOnSubmit = (form) => {
    form.addEventListener("submit", () => {
      form.classList.add("is-submitting");
      const buttons = form.querySelectorAll("button");
      buttons.forEach((button) => {
        button.setAttribute("disabled", "disabled");
      });
    });
  };

  document.querySelectorAll("form[data-disable-on-submit]").forEach((formNode) => {
    if (formNode instanceof HTMLFormElement) {
      lockFormOnSubmit(formNode);
    }
  });

  const form = document.getElementById("review-form");
  if (!form || !(form instanceof HTMLFormElement)) {
    return;
  }

  const statusInput = document.getElementById("review-status");
  const actionButtons = form.querySelectorAll("button[data-review-status]");
  const detailLink = document.querySelector("[data-open-detail]");
  const queueLinks = Array.from(document.querySelectorAll("[data-queue-link]"));

  const submitWithStatus = (status) => {
    if (statusInput instanceof HTMLInputElement) {
      statusInput.value = status;
    }

    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
      return;
    }

    form.submit();
  };

  const navigateQueue = (step) => {
    if (queueLinks.length === 0) {
      return;
    }

    const activeIndex = queueLinks.findIndex((link) => link.classList.contains("active"));
    const currentIndex = activeIndex >= 0 ? activeIndex : 0;
    const nextIndex = (currentIndex + step + queueLinks.length) % queueLinks.length;

    if (nextIndex !== currentIndex) {
      window.location.assign(queueLinks[nextIndex].href);
    }
  };

  actionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const status = button.getAttribute("data-review-status") || "PENDING";
      if (statusInput instanceof HTMLInputElement) {
        statusInput.value = status;
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
      return;
    }

    const key = event.key.toLowerCase();
    if (key === "y") {
      event.preventDefault();
      submitWithStatus("QUALIFIED");
      return;
    }

    if (key === "n") {
      event.preventDefault();
      submitWithStatus("NOT_QUALIFIED");
      return;
    }

    if (key === "s") {
      event.preventDefault();
      submitWithStatus("PENDING");
      return;
    }

    if (key === "j") {
      event.preventDefault();
      navigateQueue(-1);
      return;
    }

    if (key === "k") {
      event.preventDefault();
      navigateQueue(1);
      return;
    }

    if (key === "enter" && detailLink instanceof HTMLAnchorElement) {
      event.preventDefault();
      window.location.assign(detailLink.href);
    }
  });
})();
