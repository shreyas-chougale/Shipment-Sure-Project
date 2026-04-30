/**
 * script.js – ShipmentSure
 * Fetch-based form submission, animated result display.
 * No frameworks. No emojis. Integrates with Flask /predict endpoint.
 */

"use strict";

document.addEventListener("DOMContentLoaded", () => {

  /* ── DOM ─────────────────────────────────────────────────────── */
  const form       = document.getElementById("predict-form");
  const submitBtn  = document.getElementById("submit-btn");
  const spinner    = document.getElementById("spinner");
  const btnIcon    = document.getElementById("btn-icon");
  const btnText    = document.getElementById("btn-text");
  const errorBox   = document.getElementById("error-box");
  const resultCard = document.getElementById("result-card");

  const ontimePct   = document.getElementById("ontime-pct");
  const ontimeBar   = document.getElementById("ontime-bar");
  const ontimeDesc  = document.getElementById("ontime-desc");
  const ontimeBadge = document.getElementById("ontime-badge");
  const ontimeBText = document.getElementById("ontime-badge-text");

  const delayPct   = document.getElementById("delay-pct");
  const delayBar   = document.getElementById("delay-bar");
  const delayDesc  = document.getElementById("delay-desc");
  const delayBadge = document.getElementById("delay-badge");
  const delayBText = document.getElementById("delay-badge-text");

  /* ── Form submit ─────────────────────────────────────────────── */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    const payload = collectPayload();
    if (!payload) return;

    setLoading(true);

    try {
      const res  = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error || `Server error (${res.status})`);
        return;
      }

      renderResult(data);

    } catch (_) {
      showError("Cannot reach the server. Make sure Flask is running on port 5000.");
    } finally {
      setLoading(false);
    }
  });

  /* ── Collect & validate payload ──────────────────────────────── */
  function collectPayload() {
    const fd = new FormData(form);

    const fields = {
      Customer_rating:     parseFloat(fd.get("Customer_rating")),
      Cost_of_the_Product: parseFloat(fd.get("Cost_of_the_Product")),
      Prior_purchases:     parseInt(fd.get("Prior_purchases"), 10),
      Discount_offered:    parseFloat(fd.get("Discount_offered")),
      Weight_in_gms:       parseFloat(fd.get("Weight_in_gms")),
      Mode_of_Shipment:    fd.get("Mode_of_Shipment"),
      Product_importance:  fd.get("Product_importance"),
      Warehouse_block:     fd.get("Warehouse_block"),
    };

    for (const [key, val] of Object.entries(fields)) {
      const empty = val === null || val === "" ||
                    (typeof val === "number" && isNaN(val));
      if (empty) {
        showError(`Please fill in: ${key.replaceAll("_", " ")}`);
        return null;
      }
    }
    return fields;
  }

  /* ── Render result ───────────────────────────────────────────── */
  function renderResult(data) {
    const pctOn  = Math.round(data.probability_on_time * 100);
    const pctDel = Math.round(data.probability_delayed * 100);

    /* Numbers */
    ontimePct.textContent = `${pctOn}%`;
    delayPct.textContent  = `${pctDel}%`;

    /* Descriptions */
    ontimeDesc.textContent = pctOn >= 60
      ? "This shipment has a high chance of being delivered on time."
      : "There is some risk that this shipment may not arrive on time.";

    delayDesc.textContent = pctDel < 35
      ? "There is a low risk of delay for this shipment."
      : pctDel < 60
        ? "There is a moderate delay risk for this shipment."
        : "This shipment has a high risk of being delayed.";

    /* Badges */
    ontimeBText.textContent = pctOn >= 60 ? "On-Time Delivery Likely" : "Delivery At Risk";
    delayBText.textContent  = pctDel < 35 ? "Low Delay Risk"
                            : pctDel < 60 ? "Moderate Delay Risk"
                            : "High Delay Risk";

    /* Animate bars */
    ontimeBar.style.width = "0%";
    delayBar.style.width  = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => {
        ontimeBar.style.width = `${pctOn}%`;
        delayBar.style.width  = `${pctDel}%`;
      }, 60);
    });

    /* Show card */
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ── Helpers ─────────────────────────────────────────────────── */
  function setLoading(on) {
    submitBtn.disabled = on;
    spinner.classList.toggle("hidden", !on);
    btnIcon.classList.toggle("hidden", on);
    btnText.textContent = on ? "Predicting..." : "Predict Delivery Status";
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

});
