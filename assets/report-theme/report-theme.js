(() => {
  const key = "project-report-theme";
  let theme;
  try {
    theme = localStorage.getItem(key);
  } catch {}
  if (theme !== "light" && theme !== "dark") {
    theme = matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }
  document.documentElement.dataset.reportTheme = theme;
  function start() {
    const button = document.querySelector(".report-theme-toggle");
    function label() {
      const next =
        document.documentElement.dataset.reportTheme === "dark"
          ? "light"
          : "dark";
      button.textContent = next === "dark" ? "Dark" : "Light";
      button.setAttribute("aria-label", `Switch to ${next} theme`);
    }
    if (button) {
      button.hidden = false;
      label();
      button.addEventListener("click", () => {
        theme =
          document.documentElement.dataset.reportTheme === "dark"
            ? "light"
            : "dark";
        document.documentElement.dataset.reportTheme = theme;
        try {
          localStorage.setItem(key, theme);
        } catch {}
        label();
      });
    }
    document.querySelectorAll("table").forEach((table) => {
      const head = table.querySelector("tr:has(th)");
      if (!head) return;
      const labels = [...head.querySelectorAll("th")].map((cell) =>
        cell.textContent.trim(),
      );
      if (
        [...head.children].some((cell) => cell.colSpan > 1 || cell.rowSpan > 1)
      )
        return;
      table.classList.add("report-responsive-table");
      head.classList.add("report-table-head");
      table.querySelectorAll("tr").forEach((row) => {
        if (row === head) return;
        [...row.children].forEach((cell, i) => {
          if (cell.tagName === "TD" && !cell.hasAttribute("data-label"))
            cell.dataset.label = labels[i] || `Value ${i + 1}`;
        });
      });
    });
    const sidebar = document.querySelector(".sidebar");
    const main = document.querySelector("main, .with-sidebar, .main-content");
    if (sidebar && main) {
      const links = [...sidebar.querySelectorAll('a[href^="#"]')].filter(
        (link) =>
          link.hash.length > 1 &&
          document.getElementById(decodeURIComponent(link.hash.slice(1))),
      );
      const unique = [
        ...new Map(links.map((link) => [link.hash, link])).values(),
      ];
      if (unique.length) {
        const details = document.createElement("details");
        details.className = "report-contents";
        const summary = document.createElement("summary");
        summary.textContent = "Contents";
        const nav = document.createElement("nav");
        nav.setAttribute("aria-label", "Report contents");
        unique.forEach((link) => {
          const copy = document.createElement("a");
          copy.href = link.hash;
          copy.textContent = link.textContent.trim();
          copy.addEventListener("click", () => {
            details.open = false;
          });
          nav.append(copy);
        });
        details.append(summary, nav);
        main.prepend(details);
      }
    }
  }
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
  else start();
})();
