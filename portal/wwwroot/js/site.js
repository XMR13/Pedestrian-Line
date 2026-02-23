(function () {
  const form = document.getElementById("review-form");
  if (!form) {
    return;
  }

  const statusInput = document.getElementById("review-status");
  const actionButtons = form.querySelectorAll("button[data-review-status]");

  const submitWithStatus = (status) => {
    if (!statusInput) {
      return;
    }
    statusInput.value = status;
    form.submit();
  };

  actionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const status = button.getAttribute("data-review-status") || "PENDING";
      if (statusInput) {
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
    } else if (key === "n") {
      event.preventDefault();
      submitWithStatus("NOT_QUALIFIED");
    } else if (key === "s") {
      event.preventDefault();
      submitWithStatus("PENDING");
    }
  });
})();
