/**
 * ============================================================================
 * Product Review Analyzer — Frontend Application Logic
 * Academic Focus: GLiNER Sequence Labelling & Aspect-Based Sentiment Analysis
 * ============================================================================
 */

// ============================================================================
// DATASET: Verified Catalog Products & Genuine Customer Reviews (with ASINs)
// ============================================================================
var productsData = [];

async function loadProductsData() {
    try {
        const response = await fetch('/static/data_products.json');
        if (response.ok) {
            productsData = await response.json();
        } else {
            console.error('Failed to load /static/data_products.json:', response.status);
        }
    } catch (err) {
        console.error('Error fetching products dataset:', err);
    }
}

// ============================================================================
// CENTRAL APPLICATION STATE (Strictly Single Source of Truth)
// ============================================================================
const appState = {
    selectedProductId: null,      // ASIN identifier of selected product
    selectedProduct: null,        // Full product object
    activeCategory: "All",        // Active category filter
    searchQuery: "",              // Search input string
    sentimentFilter: "all",       // Review sentiment filter: all, positive, negative, mixed
    activeReviewId: null,         // Currently selected review ID
    activeReviewText: "",         // Text currently being analyzed
    analysisResult: null,         // Live JSON response from Flask /api/analyze
    bioViewMode: "table",         // 'table' or 'chips'
    isAnalyzing: false            // Loading status flag
};

// ============================================================================
// HELPER FUNCTIONS: HTML Escaping, Star Icons, Label Normalization
// ============================================================================
function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function normalizeLabelClass(label) {
    if (!label) return "o-tag";
    const cleaned = label.toLowerCase().trim();
    if (cleaned === "o") return "o-tag";
    if (cleaned.includes("brand")) return cleaned.startsWith("b-") ? "b-brand" : "i-brand";
    if (cleaned.includes("product")) return cleaned.startsWith("b-") ? "b-product" : "i-product";
    if (cleaned.includes("feature")) return cleaned.startsWith("b-") ? "b-feature" : "i-feature";
    if (cleaned.includes("issue")) return cleaned.startsWith("b-") ? "b-issue" : "i-issue";
    if (cleaned.includes("opinion")) return cleaned.startsWith("b-") ? "b-opinion" : "i-opinion";
    return "o-tag";
}

function getEntityType(label) {
    if (!label || label.toUpperCase() === "O") return "Outside (O)";
    const upper = label.toUpperCase();
    if (upper.includes("BRAND")) return "BRAND";
    if (upper.includes("PRODUCT")) return "PRODUCT";
    if (upper.includes("FEATURE")) return "FEATURE";
    if (upper.includes("ISSUE")) return "ISSUE";
    if (upper.includes("OPINION")) return "OPINION";
    return upper;
}

function getBioPrefix(label) {
    if (!label || label.toUpperCase() === "O") return "Outside";
    if (label.toUpperCase().startsWith("B-")) return "Begin (B)";
    if (label.toUpperCase().startsWith("I-")) return "Inside (I)";
    return "Outside";
}

function renderStars(rating) {
    const full = Math.floor(rating);
    const half = rating - full >= 0.5 ? 1 : 0;
    const empty = 5 - full - half;
    return "★".repeat(full) + (half ? "★" : "") + "☆".repeat(empty);
}

function getProductSvg(type) {
    switch (type) {
        case "headphones":
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/></svg>`;
        case "phone":
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#0891b2" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>`;
        case "watch":
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="7"/><polyline points="12 9 12 12 13.5 13.5"/><path d="M16.51 17.35l-.35 3.83a2 2 0 0 1-2 1.82H9.83a2 2 0 0 1-2-1.82l-.35-3.83m.01-10.7l.35-3.83A2 2 0 0 1 9.83 1h4.35a2 2 0 0 1 2 1.82l.35 3.83"/></svg>`;
        case "laptop":
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M2 20h20"/></svg>`;
        case "cooker":
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 10h12v9a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3v-9z"/><path d="M12 4v3"/><path d="M8 7h8"/><circle cx="12" cy="15" r="2"/></svg>`;
        default:
            return `<svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>`;
    }
}

function getProductThumbIcon(type) {
    switch (type) {
        case "headphones": return "🎧";
        case "phone": return "📱";
        case "watch": return "⌚";
        case "laptop": return "💻";
        case "cooker": return "🍲";
        default: return "📦";
    }
}

// ============================================================================
// STALE DATA CLEANUP (MANDATORY REQUIREMENT)
// Ensures that NO fields from a previously selected product remain on screen.
// ============================================================================
function clearStaleData() {
    // 1. Clear Product Info Card
    document.getElementById("productName").textContent = "Loading Product...";
    document.getElementById("productBrand").textContent = "...";
    document.getElementById("productAsin").textContent = "...";
    document.getElementById("productCategory").textContent = "...";
    document.getElementById("productType").textContent = "...";
    document.getElementById("productRatingNum").textContent = "-.-";
    document.getElementById("productStarRating").textContent = "☆☆☆☆☆";
    document.getElementById("productReviewCount").textContent = "(0 reviews)";
    document.getElementById("productDescription").textContent = "Loading details...";
    document.getElementById("productVisualContainer").innerHTML = `<div class="nlp-spinner" style="width:28px;height:28px;margin:auto;"></div>`;

    // 2. Clear Quick Sentiment Widget
    const widgetFace = document.getElementById("widgetFace");
    widgetFace.textContent = "•";
    widgetFace.className = "widget-face";
    document.getElementById("widgetSentimentLabel").textContent = "Calculating...";
    document.getElementById("widgetSentimentScore").textContent = "Pending analysis";

    // 3. Clear Reviews Container & Active Review Input
    document.getElementById("productReviewsContainer").innerHTML = `
        <div style="padding:1.5rem;text-align:center;color:#64748b;font-size:0.85rem;">
            Loading verified reviews...
        </div>`;
    document.getElementById("reviewInput").value = "";
    document.getElementById("statusMessage").textContent = "";

    // 4. Clear Highlighted Review Box
    document.getElementById("annotatedReviewBox").innerHTML = `
        <span class="placeholder-text">Loading review text for sequence labelling...</span>`;

    // 5. Clear BIO Metrics, Table & Chips
    document.getElementById("statTotalTokens").textContent = "0";
    document.getElementById("statEntityTokens").textContent = "0";
    document.getElementById("statOutsideTokens").textContent = "0";
    document.getElementById("statEntitySpans").textContent = "0";

    document.getElementById("bioTableBody").innerHTML = `
        <tr>
            <td colspan="5" class="empty-table-cell">Analyzing product review with GLiNER...</td>
        </tr>`;
    document.getElementById("bioChipsContainer").innerHTML = `
        <span class="placeholder-text">Tokens and BIO labels will appear here.</span>`;

    // 6. Clear Entity Extraction Groups
    ["Brand", "Product", "Feature", "Issue", "Opinion"].forEach(grp => {
        const countEl = document.getElementById(`count${grp}`);
        const listEl = document.getElementById(`list${grp}`);
        if (countEl) countEl.textContent = "0";
        if (listEl) listEl.innerHTML = `<span class="no-entity-hint">None detected</span>`;
    });

    // 7. Clear Aspect-Opinion Pairs & Aspect Sentiment
    document.getElementById("aspectPairsContainer").innerHTML = `
        <span class="placeholder-text">Aspect-opinion pairs will appear after analysis.</span>`;
    document.getElementById("aspectSentimentList").innerHTML = `
        <span class="placeholder-text">Aspect sentiment breakdown will appear here.</span>`;

    // 8. Clear Overall Result & Explanation
    const overallCard = document.getElementById("overallSentimentCard");
    overallCard.className = "overall-sentiment-display";
    document.getElementById("overallResultIcon").textContent = "•";
    document.getElementById("overallResultLabel").textContent = "PENDING";
    document.getElementById("overallResultScore").textContent = "Score: --";
    document.getElementById("explanationText").textContent = "Analyzing review with GLiNER...";

    // 9. Hide Error Banner
    hideError();
}

// ============================================================================
// UI STATE HELPERS: Loading & Error Management
// ============================================================================
function showLoading(text = "Analyzing Review with GLiNER...") {
    appState.isAnalyzing = true;
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) {
        overlay.querySelector(".spinner-text").textContent = text;
        overlay.style.display = "flex";
    }
    const btn = document.getElementById("analyzeBtn");
    if (btn) btn.disabled = true;
    document.getElementById("statusMessage").textContent = text;
}

function hideLoading() {
    appState.isAnalyzing = false;
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) overlay.style.display = "none";
    const btn = document.getElementById("analyzeBtn");
    if (btn) btn.disabled = false;
}

function showError(msg) {
    const banner = document.getElementById("errorBanner");
    const msgText = document.getElementById("errorMessageText");
    if (banner && msgText) {
        msgText.textContent = msg;
        banner.style.display = "flex";
    }
    document.getElementById("statusMessage").textContent = msg;
}

function hideError() {
    const banner = document.getElementById("errorBanner");
    if (banner) banner.style.display = "none";
}

// ============================================================================
// 2. PRODUCT SELECTION & SIDEBAR RENDERING
// ============================================================================
function renderSidebarProducts() {
    const container = document.getElementById("productList");
    const query = appState.searchQuery.trim().toLowerCase();
    const cat = appState.activeCategory;

    const filtered = productsData.filter(product => {
        const matchesCategory = (cat === "All" || product.brand === cat || product.category === cat);
        const matchesSearch = !query ||
            product.name.toLowerCase().includes(query) ||
            product.brand.toLowerCase().includes(query) ||
            product.asin.toLowerCase().includes(query) ||
            product.type.toLowerCase().includes(query);

        return matchesCategory && matchesSearch;
    });

    if (filtered.length === 0) {
        container.innerHTML = `
            <div style="padding:1.5rem;text-align:center;color:#64748b;font-size:0.85rem;">
                No matching products found.
            </div>`;
        return;
    }

    container.innerHTML = filtered.map(product => {
        const isSelected = (product.asin === appState.selectedProductId);
        const thumb = product.imageUrl 
            ? `<img src="${escapeHtml(product.imageUrl)}" class="sidebar-item-thumb-img" alt="${escapeHtml(product.name)}" onerror="this.onerror=null;this.parentElement.textContent='📱';">` 
            : getProductThumbIcon(product.iconType || "phone");

        return `
            <div class="product-item-card ${isSelected ? 'selected' : ''}" 
                 data-asin="${escapeHtml(product.asin)}"
                 role="option" 
                 aria-selected="${isSelected}">
                <div class="product-item-thumb">
                    ${thumb}
                </div>
                <div class="product-item-info">
                    <div class="product-item-name" title="${escapeHtml(product.name)}">${escapeHtml(product.name)}</div>
                    <div class="product-item-meta">
                        <span>${escapeHtml(product.brand)}</span>
                        <span class="asin-tag">${escapeHtml(product.asin)}</span>
                    </div>
                    <div class="product-item-rating">
                        <span>${renderStars(product.rating)}</span>
                        <span class="product-item-reviews-count">(${product.reviewsCount})</span>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

// ============================================================================
// PRODUCT SELECTION WORKFLOW (CRITICAL SYNCHRONIZATION FIX)
// When selecting Product A, EVERYTHING updates to Product A.
// When switching to Product B, EVERYTHING updates to Product B.
// ============================================================================
async function selectProduct(asin) {
    const product = productsData.find(p => p.asin === asin);
    if (!product) return;

    // 1. Immediately wipe all stale data from previous product
    clearStaleData();

    // 2. Set new central state
    appState.selectedProductId = product.asin;
    appState.selectedProduct = product;

    // 3. Highlight sidebar active item
    renderSidebarProducts();

    // 4. Render Product Information Card
    renderProductInfo(product);

    // 5. Render Verified Reviews for this selected product
    renderProductReviews(product);

    // 6. Automatically select and analyze the first review belonging to this product
    if (product.reviews && product.reviews.length > 0) {
        const firstReview = product.reviews[0];
        await selectReview(firstReview);
    } else {
        document.getElementById("reviewInput").value = "";
        document.getElementById("statusMessage").textContent = "No verified reviews for this product.";
    }
}

// ============================================================================
// 3. RENDER PRODUCT INFORMATION CARD
// ============================================================================
function renderProductInfo(product) {
    document.getElementById("productName").textContent = product.name;
    document.getElementById("productBrand").textContent = product.brand;
    document.getElementById("productAsin").textContent = product.asin;
    document.getElementById("productCategory").textContent = product.category;
    document.getElementById("productType").textContent = product.type;
    document.getElementById("productRatingNum").textContent = product.rating.toFixed(1);
    document.getElementById("productStarRating").textContent = renderStars(product.rating);
    document.getElementById("productReviewCount").textContent = `(${product.reviewsCount.toLocaleString()} verified reviews)`;
    document.getElementById("productDescription").textContent = product.description;

    const visual = document.getElementById("productVisualContainer");
    if (visual) {
        if (product.imageUrl) {
            visual.innerHTML = `<img src="${escapeHtml(product.imageUrl)}" alt="${escapeHtml(product.name)}" class="product-real-img" onerror="this.onerror=null;this.parentElement.innerHTML=getProductSvg('${product.brand}');">`;
        } else {
            visual.innerHTML = getProductSvg(product.brand || product.iconType);
        }
    }
}

// ============================================================================
// 4. RENDER PRODUCT'S VERIFIED REVIEWS
// ============================================================================
function renderProductReviews(product) {
    const container = document.getElementById("productReviewsContainer");
    const filter = appState.sentimentFilter;

    if (!product.reviews || product.reviews.length === 0) {
        container.innerHTML = `
            <div style="padding:1.5rem;text-align:center;color:#64748b;font-size:0.85rem;">
                No reviews available for this product.
            </div>`;
        return;
    }

    const filteredReviews = product.reviews.filter(rev => {
        if (filter === "all") return true;
        return rev.sentiment.toLowerCase() === filter.toLowerCase();
    });

    if (filteredReviews.length === 0) {
        container.innerHTML = `
            <div style="padding:1.5rem;text-align:center;color:#64748b;font-size:0.85rem;">
                No reviews match "${filter}" sentiment filter.
            </div>`;
        return;
    }

    container.innerHTML = filteredReviews.map(rev => {
        const isActive = (rev.id === appState.activeReviewId);
        return `
            <div class="review-card-item ${isActive ? 'active' : ''}" data-rev-id="${escapeHtml(rev.id)}">
                <div class="review-card-meta">
                    <div>
                        <span class="review-card-stars">${renderStars(rev.stars)}</span>
                        <span class="review-card-author">&nbsp;${escapeHtml(rev.author)} • ${escapeHtml(rev.date)}</span>
                    </div>
                    <span class="review-card-badge ${rev.sentiment}">${escapeHtml(rev.sentiment)}</span>
                </div>
                <div class="review-card-text" title="${escapeHtml(rev.text)}">
                    ${escapeHtml(rev.text)}
                </div>
            </div>
        `;
    }).join("");

    // Attach click listeners to review cards
    container.querySelectorAll(".review-card-item").forEach(card => {
        card.addEventListener("click", () => {
            const revId = card.getAttribute("data-rev-id");
            const rev = product.reviews.find(r => r.id === revId);
            if (rev) {
                selectReview(rev);
            }
        });
    });
}

// ============================================================================
// SELECT & LOAD A REVIEW
// ============================================================================
async function selectReview(reviewObj) {
    appState.activeReviewId = reviewObj.id;
    appState.activeReviewText = reviewObj.text;

    // Highlight selected review card in the list
    document.querySelectorAll(".review-card-item").forEach(card => {
        if (card.getAttribute("data-rev-id") === reviewObj.id) {
            card.classList.add("active");
        } else {
            card.classList.remove("active");
        }
    });

    // Populate the review textarea
    const input = document.getElementById("reviewInput");
    if (input) input.value = reviewObj.text;

    // Trigger live analysis via existing Flask API
    await runAnalysis(reviewObj.text);
}

// ============================================================================
// 11. API INTEGRATION: CALL EXISTING FLASK BACKEND (/api/analyze)
// Communicates with Flask GLiNER + Sequence Labelling + Sentiment pipeline
// ============================================================================
async function runAnalysis(reviewText) {
    const text = (reviewText || "").trim();

    if (!text) {
        showError("Please enter or select a product review for analysis.");
        return;
    }

    hideError();
    showLoading("Analyzing review with GLiNER...");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ review: text })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Review analysis failed. Backend returned an error.");
        }

        // Store active result in central state
        appState.analysisResult = data;

        // Render all analysis views using the live backend response
        renderAnalysisOutput(data);

        document.getElementById("statusMessage").textContent = "GLiNER analysis completed successfully.";

    } catch (err) {
        console.error("API error during GLiNER analysis:", err);
        showError(`Analysis failed: ${err.message}`);
    } finally {
        hideLoading();
    }
}

// ============================================================================
// RENDER ALL NLP & SEQUENCE LABELLING SECTIONS
// ============================================================================
function renderAnalysisOutput(result) {
    // 7. Render Highlighted Review Text
    renderAnnotatedReview(result.review, result.entities || []);

    // 5. Render Sequence Labelling (BIO Table & Chips)
    renderSequenceLabelling(result.bio_sequence || [], result.tokens || [], result.entities || []);

    // 6. Render Extracted Entities by Group
    renderExtractedEntities(result.entities || []);

    // 8. Render Aspect-Opinion Pairs
    renderAspectOpinionPairs(result.aspect_sentiments || []);

    // 9. Render Aspect-Level Sentiment
    renderAspectSentiment(result.aspect_sentiments || []);

    // 10. Render Overall Sentiment & Academic Explanation
    renderOverallSentiment(result.overall_sentiment || {}, result.explanation || "");
}

// ============================================================================
// 7. REVIEW HIGHLIGHTING (ANNOTATED REVIEW)
// Uses backend character offsets (start, end) to highlight entity spans safely
// ============================================================================
function renderAnnotatedReview(text, entities) {
    const container = document.getElementById("annotatedReviewBox");

    if (!text) {
        container.innerHTML = `<span class="placeholder-text">No review text to display.</span>`;
        return;
    }

    if (!entities || entities.length === 0) {
        container.innerHTML = `<div>${escapeHtml(text)}</div>`;
        return;
    }

    // Filter valid spans and sort descending by start offset
    const validSpans = entities
        .filter(e => e.start !== undefined && e.end !== undefined && e.start < e.end)
        .sort((a, b) => b.start - a.start);

    // Keep only non-overlapping spans from right to left
    const nonOverlapping = [];
    let lastStart = Infinity;
    for (const ent of validSpans) {
        if (ent.end <= lastStart) {
            nonOverlapping.push(ent);
            lastStart = ent.start;
        }
    }

    let html = escapeHtml(text);

    // Apply right-to-left replacements on raw text
    let formattedText = text;
    for (const ent of nonOverlapping) {
        const originalSpan = escapeHtml(text.slice(ent.start, ent.end));
        const cls = normalizeLabelClass(ent.label);
        const labelUpper = escapeHtml((ent.label || "").toUpperCase().replace("B-", "").replace("I-", ""));
        const scoreTag = ent.score !== undefined ? ` (${Math.round(ent.score * 100)}%)` : "";

        const highlightedMarkup = `<mark class="entity-highlight ${cls}" title="${labelUpper}${scoreTag}">${originalSpan}<span class="tag-sub">${labelUpper}</span></mark>`;

        formattedText = formattedText.slice(0, ent.start) + highlightedMarkup + formattedText.slice(ent.end);
    }

    container.innerHTML = `<div>${formattedText}</div>`;
}

// ============================================================================
// 5. SEQUENCE LABELLING SECTION (MOST IMPORTANT ACADEMIC COMPONENT)
// Token-by-Token BIO table & interactive token chips
// ============================================================================
function renderSequenceLabelling(bioSequence, tokens, entities) {
    const tableBody = document.getElementById("bioTableBody");
    const chipsContainer = document.getElementById("bioChipsContainer");

    // Metrics Calculation
    const totalTokens = bioSequence.length || tokens.length || 0;
    let entityTokensCount = 0;
    let outsideTokensCount = 0;

    bioSequence.forEach(item => {
        const lbl = (item.label || "O").toUpperCase();
        if (lbl === "O") {
            outsideTokensCount++;
        } else {
            entityTokensCount++;
        }
    });

    document.getElementById("statTotalTokens").textContent = totalTokens;
    document.getElementById("statEntityTokens").textContent = entityTokensCount;
    document.getElementById("statOutsideTokens").textContent = outsideTokensCount;
    document.getElementById("statEntitySpans").textContent = entities ? entities.length : 0;

    // If no sequence data returned
    if (!bioSequence || bioSequence.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="5" class="empty-table-cell">No tokens returned by sequence labelling model.</td>
            </tr>`;
        chipsContainer.innerHTML = `<span class="placeholder-text">No tokens returned.</span>`;
        return;
    }

    // 1. Render Table View
    tableBody.innerHTML = bioSequence.map((item, idx) => {
        const token = item.token || "";
        const label = item.label || "O";
        const badgeClass = normalizeLabelClass(label);
        const entityType = getEntityType(label);
        const bioPrefix = getBioPrefix(label);

        return `
            <tr>
                <td style="color:#94a3b8; font-family:var(--font-mono); font-size:0.75rem;">${idx + 1}</td>
                <td class="token-cell">${escapeHtml(token)}</td>
                <td>
                    <span class="bio-badge ${badgeClass}">${escapeHtml(label)}</span>
                </td>
                <td style="font-weight:500; font-size:0.82rem;">${escapeHtml(entityType)}</td>
                <td style="color:#64748b; font-size:0.8rem;">${escapeHtml(bioPrefix)}</td>
            </tr>
        `;
    }).join("");

    // 2. Render Token Chips View
    chipsContainer.innerHTML = bioSequence.map(item => {
        const token = item.token || "";
        const label = item.label || "O";
        const badgeClass = normalizeLabelClass(label);

        return `
            <div class="token-chip" title="Token: ${escapeHtml(token)} | BIO: ${escapeHtml(label)}">
                <span class="token-chip-text">${escapeHtml(token)}</span>
                <span class="token-chip-label bio-badge ${badgeClass}">${escapeHtml(label)}</span>
            </div>
        `;
    }).join("");
}

// ============================================================================
// 6. EXTRACTED ENTITIES SECTION (GROUPED BY CATEGORY)
// Groups: BRAND, PRODUCT, FEATURE, ISSUE, OPINION
// ============================================================================
function renderExtractedEntities(entities) {
    const groups = {
        Brand: [],
        Product: [],
        Feature: [],
        Issue: [],
        Opinion: []
    };

    entities.forEach(ent => {
        const lbl = (ent.label || "").toUpperCase();
        let targetGrp = null;

        if (lbl.includes("BRAND")) targetGrp = "Brand";
        else if (lbl.includes("PRODUCT")) targetGrp = "Product";
        else if (lbl.includes("FEATURE")) targetGrp = "Feature";
        else if (lbl.includes("ISSUE")) targetGrp = "Issue";
        else if (lbl.includes("OPINION")) targetGrp = "Opinion";

        if (targetGrp) {
            groups[targetGrp].push(ent);
        }
    });

    Object.keys(groups).forEach(grp => {
        const countEl = document.getElementById(`count${grp}`);
        const listEl = document.getElementById(`list${grp}`);
        const items = groups[grp];

        if (countEl) countEl.textContent = items.length;

        if (listEl) {
            if (items.length === 0) {
                listEl.innerHTML = `<span class="no-entity-hint">None detected</span>`;
            } else {
                listEl.innerHTML = items.map(item => {
                    const text = item.text || item.token || "";
                    const score = item.score !== undefined ? `${Math.round(item.score * 100)}%` : "";
                    return `
                        <div class="entity-chip">
                            <span>${escapeHtml(text)}</span>
                            ${score ? `<span class="score-tag">${score}</span>` : ""}
                        </div>
                    `;
                }).join("");
            }
        }
    });
}

// ============================================================================
// 8. ASPECT-OPINION ANALYSIS
// Displays paired Aspects and Opinions with visual connectors
// ============================================================================
function renderAspectOpinionPairs(aspectSentiments) {
    const container = document.getElementById("aspectPairsContainer");

    if (!aspectSentiments || aspectSentiments.length === 0) {
        container.innerHTML = `
            <div style="padding:1rem;color:#64748b;font-size:0.85rem;font-style:italic;">
                No aspect-opinion pairs identified in this review.
            </div>`;
        return;
    }

    container.innerHTML = aspectSentiments.map(item => {
        const aspect = item.aspect || "Aspect";
        const opinion = item.opinion || "General Context";
        const rawSentiment = String(item.sentiment || item.label || "NEUTRAL").toUpperCase();
        const sentimentCls = rawSentiment.toLowerCase();
        const reason = item.reason ? `<div style="font-size:0.75rem;color:#64748b;margin-top:0.25rem;">${escapeHtml(item.reason)}</div>` : "";

        return `
            <div class="aspect-pair-item">
                <div style="flex:1;">
                    <div class="aspect-linkage">
                        <span class="aspect-tag-box">${escapeHtml(aspect)}</span>
                        <span class="pair-arrow">──▶</span>
                        <span class="opinion-tag-box">${escapeHtml(opinion)}</span>
                    </div>
                    ${reason}
                </div>
                <span class="aspect-sentiment-badge ${sentimentCls}">${escapeHtml(rawSentiment)}</span>
            </div>
        `;
    }).join("");
}

// ============================================================================
// 9. ASPECT SENTIMENT BREAKDOWN
// ============================================================================
function renderAspectSentiment(aspectSentiments) {
    const container = document.getElementById("aspectSentimentList");

    if (!aspectSentiments || aspectSentiments.length === 0) {
        container.innerHTML = `
            <div style="padding:1rem;color:#64748b;font-size:0.85rem;font-style:italic;">
                No aspect-level sentiment identified.
            </div>`;
        return;
    }

    container.innerHTML = aspectSentiments.map(item => {
        const aspect = item.aspect || "Aspect";
        const rawSentiment = String(item.sentiment || item.label || "NEUTRAL").toUpperCase();
        const sentimentCls = rawSentiment.toLowerCase();
        const score = item.score !== undefined ? `${Math.round(Math.abs(Number(item.score)) * 100)}%` : "";

        return `
            <div class="aspect-sentiment-card">
                <span class="aspect-title">${escapeHtml(aspect)}</span>
                <div class="aspect-meta">
                    ${score ? `<span style="font-size:0.75rem;color:#64748b;font-family:var(--font-mono);">${score}</span>` : ""}
                    <span class="aspect-sentiment-badge ${sentimentCls}">${escapeHtml(rawSentiment)}</span>
                </div>
            </div>
        `;
    }).join("");
}

// ============================================================================
// 10. OVERALL RESULT & ACADEMIC EXPLANATION
// ============================================================================
function renderOverallSentiment(overall, explanation) {
    const rawLabel = String(
        overall.label || overall.sentiment || "NEUTRAL"
    ).toUpperCase();

    const sentimentCls = rawLabel.toLowerCase();
    const scoreVal = overall.score !== undefined ? Number(overall.score) : null;
    const scoreText = scoreVal !== null ? `Score: ${scoreVal.toFixed(2)}` : "Aspect Synthesized";

    // Update Overall Result Card
    const overallCard = document.getElementById("overallSentimentCard");
    overallCard.className = `overall-sentiment-display ${sentimentCls}`;

    const iconEl = document.getElementById("overallResultIcon");
    iconEl.textContent = (sentimentCls === "positive") ? "☺" :
                         (sentimentCls === "negative") ? "☹" :
                         (sentimentCls === "mixed") ? "◐" : "•";

    document.getElementById("overallResultLabel").textContent = rawLabel;
    document.getElementById("overallResultScore").textContent = scoreText;

    // Update Quick Widget in Product Info Card as well
    const widgetFace = document.getElementById("widgetFace");
    widgetFace.textContent = iconEl.textContent;
    widgetFace.className = `widget-face ${sentimentCls}`;
    document.getElementById("widgetSentimentLabel").textContent = rawLabel;
    document.getElementById("widgetSentimentScore").textContent = scoreText;

    // Update Breakdown if counts available
    if (overall.positive_count !== undefined && overall.negative_count !== undefined) {
        document.getElementById("widgetBreakdown").innerHTML = `
            <span>+${overall.positive_count} Pos</span>
            <span>•</span>
            <span>-${overall.negative_count} Neg</span>
            <span>•</span>
            <span>${overall.neutral_count || 0} Neu</span>
        `;
    }

    // Update Explanation text directly from backend
    const explanationEl = document.getElementById("explanationText");
    const reason = explanation || overall.explanation || "Sentiment was synthesized from aspect-level polarities by GLiNER and the sentiment analyzer.";
    explanationEl.textContent = reason;
}

// ============================================================================
// EVENT LISTENERS & INITIALIZATION
// ============================================================================
function initEventListeners() {
    // Event delegation for product sidebar list
    const productListEl = document.getElementById("productList");
    if (productListEl) {
        productListEl.addEventListener("click", (e) => {
            const card = e.target.closest(".product-item-card");
            if (card) {
                const asin = card.getAttribute("data-asin");
                if (asin && asin !== appState.selectedProductId) {
                    selectProduct(asin);
                }
            }
        });
    }
    // 1. Category filter tabs
    const catTabs = document.querySelectorAll("#categoryTabs .cat-tab");
    catTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            catTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            appState.activeCategory = tab.getAttribute("data-category") || "All";
            renderSidebarProducts();
        });
    });

    // 2. Product Search input & clear button
    const searchInput = document.getElementById("productSearch");
    const clearSearchBtn = document.getElementById("clearSearchBtn");

    searchInput.addEventListener("input", (e) => {
        appState.searchQuery = e.target.value;
        clearSearchBtn.style.display = appState.searchQuery ? "block" : "none";
        renderSidebarProducts();
    });

    clearSearchBtn.addEventListener("click", () => {
        searchInput.value = "";
        appState.searchQuery = "";
        clearSearchBtn.style.display = "none";
        renderSidebarProducts();
        searchInput.focus();
    });

    // 3. Sentiment filter for product reviews
    const filterSelect = document.getElementById("sentimentFilter");
    if (filterSelect) {
        filterSelect.addEventListener("change", (e) => {
            appState.sentimentFilter = e.target.value;
            if (appState.selectedProduct) {
                renderProductReviews(appState.selectedProduct);
            }
        });
    }

    // 4. Custom Review Analyze button
    const analyzeBtn = document.getElementById("analyzeBtn");
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", () => {
            const input = document.getElementById("reviewInput");
            const customText = input ? input.value.trim() : "";
            if (!customText) {
                showError("Please enter a product review to analyze.");
                return;
            }
            // Clear review card selection highlight since this is custom input
            appState.activeReviewId = null;
            document.querySelectorAll(".review-card-item").forEach(c => c.classList.remove("active"));
            runAnalysis(customText);
        });
    }

    // 5. Clear review button
    const clearReviewBtn = document.getElementById("clearReviewBtn");
    if (clearReviewBtn) {
        clearReviewBtn.addEventListener("click", () => {
            const input = document.getElementById("reviewInput");
            if (input) {
                input.value = "";
                input.focus();
            }
            document.getElementById("statusMessage").textContent = "";
        });
    }

    // 6. View toggle: Sequence Labelling Table vs Chips
    const viewTableBtn = document.getElementById("viewTableBtn");
    const viewChipsBtn = document.getElementById("viewChipsBtn");
    const tableView = document.getElementById("bioTableView");
    const chipsView = document.getElementById("bioChipsView");

    if (viewTableBtn && viewChipsBtn) {
        viewTableBtn.addEventListener("click", () => {
            viewTableBtn.classList.add("active");
            viewChipsBtn.classList.remove("active");
            tableView.style.display = "block";
            chipsView.style.display = "none";
            appState.bioViewMode = "table";
        });

        viewChipsBtn.addEventListener("click", () => {
            viewChipsBtn.classList.add("active");
            viewTableBtn.classList.remove("active");
            tableView.style.display = "none";
            chipsView.style.display = "block";
            appState.bioViewMode = "chips";
        });
    }
}

// ============================================================================
// INITIAL APPLICATION MOUNT
// ============================================================================
document.addEventListener("DOMContentLoaded", async () => {
    await loadProductsData();
    initEventListeners();

    // Render sidebar products
    renderSidebarProducts();

    // Select the first product initially (Sony WH-1000XM5)
    if (productsData.length > 0) {
        selectProduct(productsData[0].asin);
    }
});
