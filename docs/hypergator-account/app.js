const copyButtons = document.querySelectorAll("[data-copy-target]");

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

copyButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    const originalText = button.textContent;
    try {
      await copyText(target.textContent.trim());
      button.textContent = "已复制";
      button.classList.add("copied");
    } catch {
      button.textContent = "复制失败";
    }

    window.setTimeout(() => {
      button.textContent = originalText;
      button.classList.remove("copied");
    }, 1600);
  });
});

const checklistItems = document.querySelectorAll("[data-check]");

checklistItems.forEach((item) => {
  const key = `hypergator-guide-${item.dataset.check}`;
  try {
    item.checked = localStorage.getItem(key) === "true";
  } catch {
    item.checked = false;
  }

  item.addEventListener("change", () => {
    try {
      localStorage.setItem(key, item.checked ? "true" : "false");
    } catch {
      /* localStorage can be unavailable in strict browser modes. */
    }
  });
});

const navLinks = Array.from(document.querySelectorAll(".side-nav a"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

    if (!visible) return;

    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  {
    rootMargin: "-10% 0px -65% 0px",
    threshold: [0.1, 0.25, 0.5],
  }
);

sections.forEach((section) => observer.observe(section));
