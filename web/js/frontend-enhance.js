/* Runtime enhancement for the existing Frontend/dashboard.html UI.
 *
 * The protected HTML/CSS/images are never edited: this file adds data, detail
 * pages, streaming links, cinema links and model recommendations at runtime.
 */
(function () {
  "use strict";

  var config = window.MOVIEMATE_FRONTEND_CONFIG || {};
  var API_BASE = config.apiBaseUrl || "/api";
  var FALLBACK_POSTER = "/media/fallback/poster.svg";
  var REGION = (config.watchRegion || "IN").toUpperCase();
  var PAGE_SIZE = 60;

  var state = {
    moviesPage: 1,
    seriesPage: 1,
    moviesTotalPages: 1,
    seriesTotalPages: 1,
    trending: [],
    searchTimer: null,
    activeTitle: null,
    lastFocus: null,
    myList: readList()
  };

  var STATIC_ID_MAP = {
    "781732.jpg": "movie-781732",
    "835017.jpg": "movie-835017",
    "857570.jpg": "movie-857570",
    "875188.jpg": "movie-875188",
    "904759.jpg": "movie-904759",
    "960469.jpg": "movie-960469",
    "peakyblinders.jpg": "tv-60574"
  };

  /* ------------------------------------------------------------- utilities */
  function readList() {
    try {
      var saved = JSON.parse(localStorage.getItem("moviemate.myList") || "[]");
      return Array.isArray(saved) ? saved : [];
    } catch (error) {
      return [];
    }
  }

  function saveList() {
    try {
      localStorage.setItem("moviemate.myList", JSON.stringify(state.myList));
    } catch (error) {
      /* private browsing: the list still works for this page view */
    }
  }

  async function apiGet(path) {
    var response = await fetch(API_BASE + path, {
      headers: { Accept: "application/json" },
      cache: "no-store"
    });
    var payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok || payload.success === false) {
      throw new Error((payload.error && payload.error.message) || "Request failed (" + response.status + ").");
    }
    return payload;
  }

  async function apiPost(path, body) {
    var response = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Accept": "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store"
    });
    var payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok || payload.success === false) {
      throw new Error((payload.error && payload.error.message) || "Request failed (" + response.status + ").");
    }
    return payload;
  }

  function sendFeedback(contentId, event) {
    // Learning signals are best-effort; they never block the UI.
    apiPost("/feedback", { content_id: contentId, event: event }).catch(function () {});
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function safeUrl(url) {
    return typeof url === "string" && url.trim() ? url.trim() : FALLBACK_POSTER;
  }

  function posterImage(url, alt) {
    var image = element("img");
    image.src = safeUrl(url);
    image.alt = alt || "Poster";
    image.loading = "lazy";
    image.decoding = "async";
    image.addEventListener("error", function () {
      if (!image.src.endsWith(FALLBACK_POSTER)) image.src = FALLBACK_POSTER;
    });
    return image;
  }

  function metaLine(item, extra) {
    var parts = [item.release_year || "Year unknown"];
    if (Array.isArray(item.genres) && item.genres.length) parts.push(item.genres.slice(0, 2).join(", "));
    if (typeof item.rating === "number") parts.push("★ " + item.rating.toFixed(1));
    if (item.media_type === "tv") parts.push("Series");
    if (extra) parts.push(extra);
    return parts.join(" · ");
  }

  function addToMyListButton(contentId) {
    var button = element("button", "mm-add", state.myList.indexOf(contentId) >= 0 ? "✓" : "+");
    button.type = "button";
    button.title = state.myList.indexOf(contentId) >= 0 ? "Remove from My List" : "Add to My List";
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      var index = state.myList.indexOf(contentId);
      if (index >= 0) {
        state.myList.splice(index, 1);
        button.textContent = "+";
        button.title = "Add to My List";
        sendFeedback(contentId, "remove");
      } else {
        state.myList.unshift(contentId);
        button.textContent = "✓";
        button.title = "Remove from My List";
        sendFeedback(contentId, "save");
      }
      saveList();
      if (isVisible("page-mylist")) renderMyList();
    });
    return button;
  }

  function cardFor(item, extra, options) {
    // An article with an explicit role avoids nested interactive controls:
    // the card opens details, while the + / hide buttons are separate actions.
    var card = element("article", "card mm-card");
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", "Open details for " + (item.title || "title"));
    var poster = posterImage(item.poster_url, (item.title || "Title") + " poster");
    card.appendChild(poster);
    var title = element("strong", "mm-card-title", item.title || "Untitled");
    var meta = element("span", "mm-card-meta", metaLine(item, extra));
    card.append(title, meta, addToMyListButton(item.content_id));

    if (options && options.dismissible) {
      var hide = element("button", "mm-hide", "×");
      hide.type = "button";
      hide.title = "Show fewer like this";
      hide.setAttribute("aria-label", "Show fewer titles like " + (item.title || "this title"));
      hide.addEventListener("click", function (event) {
        event.stopPropagation();
        sendFeedback(item.content_id, "dismiss");
        card.remove();
      });
      card.appendChild(hide);
    }

    card.addEventListener("click", function () { openDetails(item.content_id); });
    card.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openDetails(item.content_id);
      }
    });
    return card;
  }

  function grid(container, items, extra) {
    var wrapper = element("div", "card-grid mm-grid");
    items.forEach(function (item) { wrapper.appendChild(cardFor(item, extra)); });
    container.appendChild(wrapper);
    return wrapper;
  }

  function sectionHeading(container, text, hint) {
    var heading = element("div", "mm-section-heading");
    heading.appendChild(element("h2", "", text));
    if (hint) heading.appendChild(element("span", "mm-hint", hint));
    container.appendChild(heading);
  }

  function stateMessage(container, text) {
    container.appendChild(element("p", "mm-message", text));
  }

  function loadingText(text, withCards) {
    var wrapper = element("div", "mm-loading", text);
    wrapper.setAttribute("role", "status");
    wrapper.setAttribute("aria-live", "polite");
    if (withCards) {
      var skeletons = element("div", "mm-skeleton");
      skeletons.setAttribute("aria-hidden", "true");
      for (var index = 0; index < 6; index += 1) skeletons.appendChild(element("span"));
      wrapper.appendChild(skeletons);
    }
    return wrapper;
  }

  function isVisible(pageId) {
    var page = document.getElementById(pageId);
    return Boolean(page && page.classList.contains("active"));
  }

  /* --------------------------------------------------------------- overlays */
  function injectStyles() {
    if (document.getElementById("mm-enhance-style")) return;
    var style = document.createElement("style");
    style.id = "mm-enhance-style";
    style.textContent = [
      ".mm-card{position:relative;cursor:pointer;overflow:hidden;text-align:left;color:#fff;height:auto;min-height:256px;padding:0;display:block;border-radius:10px;transition:transform .15s ease,border-color .15s ease}",
      ".mm-card:focus-visible{outline:3px solid #ff8a65;outline-offset:2px}",
      ".mm-card:hover{border-color:#555;transform:translateY(-2px)}",
      ".mm-card img{width:100%;height:196px;object-fit:cover;display:block;background:#202020}",
      ".mm-card-title,.mm-card-meta{display:block;padding:7px 12px 0;font-size:13px;line-height:1.3}",
      ".mm-card-meta{color:#b8b8b8;font-size:12px;padding-top:4px;padding-bottom:12px}",
      ".mm-add,.mm-hide{position:absolute;top:8px;width:44px;height:44px;border-radius:50%;border:1px solid rgba(255,255,255,.7);background:rgba(0,0,0,.78);color:#fff;font-size:22px;line-height:1;cursor:pointer;z-index:3;display:grid;place-items:center}",
      ".mm-add{right:8px}",
      ".mm-hide{left:8px;opacity:0;transition:opacity .15s ease}",
      ".mm-card:hover .mm-hide,.mm-card:focus-within .mm-hide{opacity:1}",
      ".mm-add:hover,.mm-hide:hover{background:#e50914;border-color:#e50914}",
      "@media (hover:none){.mm-hide{opacity:1}}",
      ".mm-section-heading{display:flex;align-items:baseline;gap:12px;margin:22px 0 10px}",
      ".mm-section-heading h2{font-size:1.25rem}",
      ".mm-hint,.mm-message{color:#aaa;font-size:13px}",
      ".mm-message{margin:10px 0 18px}",
      ".mm-actions{margin:16px 0 8px;display:flex;flex-wrap:wrap;gap:10px}",
      ".mm-action{min-height:44px;padding:10px 18px;border:1px solid #555;background:#1c1c1c;color:#fff;border-radius:10px;cursor:pointer;font:inherit;font-size:14px}",
      ".mm-action:hover{background:#e50914;border-color:#e50914}",
      ".mm-detail{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.88);overflow:auto;display:none}",
      ".mm-detail.open{display:block}",
      ".mm-detail-panel{max-width:1100px;margin:32px auto;background:#0f0f0f;border:1px solid #303030;border-radius:16px;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.75)}",
      ".mm-detail-close{position:sticky;top:0;float:right;margin:12px 14px 0 0;z-index:4;width:44px;height:44px;border-radius:50%;border:1px solid #666;background:#111;color:#fff;font-size:24px;cursor:pointer;display:grid;place-items:center}",
      ".mm-detail-close:hover{background:#e50914;border-color:#e50914}",
      ".mm-hero{position:relative;min-height:390px;background:#111}",
      ".mm-hero img{width:100%;height:390px;object-fit:cover;opacity:.42}",
      ".mm-hero-content{position:absolute;left:32px;bottom:28px;right:32px;max-width:760px;text-shadow:0 2px 10px #000}",
      ".mm-hero-content h2{font-size:2.4rem;margin-bottom:10px}",
      ".mm-badge{display:inline-block;padding:3px 9px;border:1px solid #e50914;border-radius:3px;color:#fff;font-size:11px;letter-spacing:.08em;margin-right:10px}",
      ".mm-detail-body{padding:28px 32px 40px}",
      ".mm-detail-meta{color:#bbb;margin-bottom:16px;font-size:14px}",
      ".mm-overview{line-height:1.65;max-width:900px;margin-bottom:18px}",
      ".mm-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}",
      ".mm-chip{background:#222;border:1px solid #333;border-radius:999px;padding:4px 11px;font-size:12px}",
      ".mm-block{margin-top:28px;border-top:1px solid #262626;padding-top:20px}",
      ".mm-block h3{font-size:1.1rem;margin-bottom:14px}",
      ".mm-provider-grid{display:flex;flex-wrap:wrap;gap:10px}",
      ".mm-provider{display:flex;align-items:center;gap:10px;background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:8px 12px;color:#fff;text-decoration:none;min-width:150px}",
      ".mm-provider:hover{border-color:#e50914}",
      ".mm-provider img{width:28px;height:28px;border-radius:5px;background:#fff}",
      ".mm-provider-name{font-size:13px}",
      ".mm-links{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}",
      ".mm-link{color:#fff;background:#222;border:1px solid #444;border-radius:4px;padding:9px 13px;text-decoration:none;font-size:13px}",
      ".mm-link:hover{background:#e50914;border-color:#e50914}",
      ".mm-note{color:#999;font-size:12px;line-height:1.5;margin-top:10px}",
      ".mm-dates{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}",
      ".mm-date{background:#1c1c1c;border:1px solid #333;padding:7px 10px;border-radius:4px;font-size:12px}",
      ".mm-loading{padding:60px;text-align:center;color:#c4c4c4;font-size:15px}",
      ".mm-skeleton{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;margin-top:14px}",
      ".mm-skeleton span{height:256px;border-radius:10px;background:linear-gradient(100deg,#1b1b1b 20%,#292929 40%,#1b1b1b 60%);background-size:220% 100%;animation:mm-shimmer 1.3s linear infinite}",
      "@keyframes mm-shimmer{to{background-position:-220% 0}}",
      "@media (max-width:700px){.mm-hero img{height:280px}.mm-hero{min-height:280px}.mm-hero-content{left:16px;bottom:18px;right:16px}.mm-hero-content h2{font-size:1.6rem}.mm-detail-body{padding:20px 16px 32px}.mm-detail-panel{margin:0;border-radius:0}}"
    ].join("\n");
    document.head.appendChild(style);
  }

  function ensureOverlay() {
    injectStyles();
    var overlay = document.getElementById("mm-detail-overlay");
    if (!overlay) {
      overlay = element("div", "mm-detail");
      overlay.id = "mm-detail-overlay";
      overlay.setAttribute("role", "dialog");
      overlay.setAttribute("aria-modal", "true");
      overlay.setAttribute("aria-labelledby", "mm-detail-title");
      var panel = element("div", "mm-detail-panel");
      panel.id = "mm-detail-panel";
      overlay.appendChild(panel);
      document.body.appendChild(overlay);
      overlay.addEventListener("click", function (event) {
        if (event.target === overlay) closeDetails();
      });
    }
    return overlay;
  }

  function closeDetails() {
    var overlay = document.getElementById("mm-detail-overlay");
    if (overlay) overlay.classList.remove("open");
    state.activeTitle = null;
    document.body.style.overflow = "";
    if (state.lastFocus && document.contains(state.lastFocus)) {
      state.lastFocus.focus();
    }
    state.lastFocus = null;
  }

  function link(label, url, className) {
    var anchor = element("a", className || "mm-link", label);
    anchor.href = url;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    return anchor;
  }

  function providerCard(provider) {
    var anchor = element("a", "mm-provider");
    anchor.href = provider.link || "https://www.justwatch.com/";
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    if (provider.logo_url) anchor.appendChild(posterImage(provider.logo_url, provider.name));
    anchor.appendChild(element("span", "mm-provider-name", provider.name));
    return anchor;
  }

  function renderAvailability(target, availability) {
    var providers = availability.providers || {};
    var labels = {
      flatrate: "Stream",
      free: "Free",
      ads: "Free with ads",
      rent: "Rent",
      buy: "Buy"
    };
    var total = 0;
    Object.keys(labels).forEach(function (key) {
      var items = providers[key] || [];
      if (!items.length) return;
      total += items.length;
      target.appendChild(element("h3", "", labels[key] + " · " + availability.region));
      var row = element("div", "mm-provider-grid");
      items.forEach(function (provider) { row.appendChild(providerCard(provider)); });
      target.appendChild(row);
    });
    if (!total) {
      target.appendChild(element("p", "mm-note", "No providers are listed for " + availability.region + " in TMDB right now."));
    }
    var links = element("div", "mm-links");
    links.appendChild(link("See all provider options", availability.provider_page_url));
    target.appendChild(links);
    target.appendChild(element("p", "mm-note", availability.note || ""));
  }

  function renderCinema(target, cinema) {
    if (!cinema || !cinema.applicable) {
      target.appendChild(element("p", "mm-note", "This is a TV series, so cinema showtimes are not applicable."));
      if (cinema && cinema.links) {
        var seriesLinks = element("div", "mm-links");
        cinema.links.forEach(function (item) { seriesLinks.appendChild(link(item.label, item.url)); });
        target.appendChild(seriesLinks);
      }
      return;
    }
    var dates = cinema.release_dates || [];
    if (dates.length) {
      var dateRow = element("div", "mm-dates");
      dates.forEach(function (item) {
        dateRow.appendChild(element("span", "mm-date", item.country + " · " + item.date + " · " + item.type));
      });
      target.appendChild(dateRow);
    }
    var cinemaLinks = element("div", "mm-links");
    (cinema.links || []).forEach(function (item) { cinemaLinks.appendChild(link(item.label, item.url)); });
    target.appendChild(cinemaLinks);
    target.appendChild(element("p", "mm-note", "TMDB supplies release dates, not live seat availability. Open a ticketing link above for current showtimes near you."));
  }

  function renderSimilar(target, results, meta) {
    if (!results.length) {
      target.appendChild(element("p", "mm-note", "No similar titles were found for this title."));
      return;
    }
    var gridNode = element("div", "card-grid mm-grid");
    results.forEach(function (item) {
      gridNode.appendChild(cardFor(
        item,
        "match " + Number(item.similarity_score).toFixed(3),
        { dismissible: true }
      ));
    });
    target.appendChild(gridNode);
    var label = meta && meta.personalization && meta.personalization.ranking ===
      "time_decayed_repetition_penalty_epsilon_greedy_bandit"
      ? "Ranked by story similarity, with recently shown titles lowered and your feedback applied."
      : "Ranked by story similarity only; this session has no personalisation yet.";
    target.appendChild(element("p", "mm-note", label + " Use × to show fewer like a title."));
  }

  function renderDetail(payload) {
    var overlay = ensureOverlay();
    var panel = document.getElementById("mm-detail-panel");
    panel.replaceChildren();

    var close = element("button", "mm-detail-close", "×");
    close.type = "button";
    close.title = "Close details";
    close.addEventListener("click", closeDetails);
    panel.appendChild(close);

    var hero = element("div", "mm-hero");
    hero.appendChild(posterImage(payload.backdrop_url || payload.poster_url, payload.title + " backdrop"));
    var heroContent = element("div", "mm-hero-content");
    heroContent.appendChild(element("span", "mm-badge", payload.media_type === "tv" ? "SERIES" : "MOVIE"));
    var heading = element("h2", "", payload.title || "Untitled");
    heading.id = "mm-detail-title";
    heroContent.appendChild(heading);
    heroContent.appendChild(element("p", "mm-detail-meta", metaLine(payload)));
    hero.appendChild(heroContent);
    panel.appendChild(hero);

    var body = element("div", "mm-detail-body");
    body.appendChild(element("p", "mm-overview", payload.overview || payload.short_description || "No overview was provided in the dataset."));

    var chips = element("div", "mm-chips");
    (payload.genres || []).forEach(function (genre) { chips.appendChild(element("span", "mm-chip", genre)); });
    if (payload.number_of_seasons) chips.appendChild(element("span", "mm-chip", payload.number_of_seasons + " season" + (payload.number_of_seasons === 1 ? "" : "s")));
    if (payload.number_of_episodes) chips.appendChild(element("span", "mm-chip", payload.number_of_episodes + " episodes"));
    if (payload.status) chips.appendChild(element("span", "mm-chip", payload.status));
    body.appendChild(chips);

    if ((payload.cast || []).length) body.appendChild(element("p", "mm-detail-meta", "Cast: " + payload.cast.slice(0, 8).join(", ")));
    if ((payload.director || []).length) body.appendChild(element("p", "mm-detail-meta", "Director: " + payload.director.join(", ")));
    if (payload.homepage) body.appendChild(link("Official website", payload.homepage, "mm-link"));

    var availability = element("section", "mm-block");
    availability.appendChild(element("h2", "", "Streaming & where to watch"));
    availability.appendChild(loadingText("Loading live providers…"));
    body.appendChild(availability);

    var cinema = element("section", "mm-block");
    cinema.appendChild(element("h2", "", "Cinema showtimes"));
    cinema.appendChild(loadingText("Loading cinema links…"));
    body.appendChild(cinema);

    var similar = element("section", "mm-block");
    similar.appendChild(element("h2", "", "More like this"));
    similar.appendChild(loadingText("Finding similar titles…", true));
    body.appendChild(similar);

    panel.appendChild(body);
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    close.focus();

    Promise.all([
      apiGet("/titles/" + encodeURIComponent(payload.content_id) + "/watch?region=" + REGION),
      apiGet("/titles/" + encodeURIComponent(payload.content_id) + "/recommendations?limit=12")
    ]).then(function (responses) {
      availability.replaceChildren(element("h2", "", "Streaming & where to watch"));
      renderAvailability(availability, responses[0].data);
      cinema.replaceChildren(element("h2", "", "Cinema showtimes"));
      renderCinema(cinema, responses[0].data.cinema);
      similar.replaceChildren(element("h2", "", "More like this"));
      renderSimilar(similar, responses[1].data, responses[1].meta);
    }).catch(function (error) {
      availability.replaceChildren(element("h2", "", "Streaming & where to watch"));
      availability.appendChild(element("p", "mm-note", "Availability could not be loaded: " + error.message));
      cinema.replaceChildren(element("h2", "", "Cinema showtimes"));
      cinema.appendChild(element("p", "mm-note", "Cinema links could not be loaded."));
      similar.replaceChildren(element("h2", "", "More like this"));
      similar.appendChild(element("p", "mm-note", "Recommendations could not be loaded: " + error.message));
    });
  }

  async function openDetails(contentId, pushState) {
    if (!contentId) return;
    state.activeTitle = contentId;
    var overlay = ensureOverlay();
    var panel = document.getElementById("mm-detail-panel");
    if (!overlay.classList.contains("open")) state.lastFocus = document.activeElement;
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    panel.replaceChildren(loadingText("Loading title details…"));
    sendFeedback(contentId, "open");

    if (pushState !== false && window.history && window.history.pushState) {
      window.history.pushState({ mmTitle: contentId }, "", "/title/" + encodeURIComponent(contentId));
    }
    try {
      var response = await apiGet("/titles/" + encodeURIComponent(contentId));
      if (state.activeTitle !== contentId) return;
      renderDetail(response.data);
    } catch (error) {
      panel.replaceChildren(
        element("div", "mm-loading", "Could not load this title: " + error.message),
        (function () {
          var back = element("button", "mm-action", "Close");
          back.addEventListener("click", closeDetails);
          return back;
        })()
      );
    }
  }

  /* ----------------------------------------------------------------- data */
  async function loadSection(pageId, path, heading, hint) {
    var container = document.getElementById(pageId);
    if (!container) return [];
    try {
      var response = await apiGet(path);
      sectionHeading(container, heading, hint);
      if (!response.data.length) stateMessage(container, "No titles are available here yet.");
      else grid(container, response.data, hint ? undefined : "top pick");
      return response.data;
    } catch (error) {
      sectionHeading(container, heading);
      stateMessage(container, "Could not load titles: " + error.message);
      return [];
    }
  }

  async function loadMoviesPage() {
    var container = document.getElementById("page-movies");
    if (!container) return;
    try {
      var response = await apiGet("/movies?page=" + state.moviesPage + "&per_page=" + PAGE_SIZE + "&sort=popularity_desc");
      if (state.moviesPage === 1) {
        sectionHeading(container, "All movies", response.meta.total + " titles available");
      }
      if (!response.data.length) {
        stateMessage(container, "No more movies to load.");
      } else {
        grid(container, response.data);
      }
      state.moviesTotalPages = response.meta.total_pages || 1;
      var button = document.getElementById("mm-more-movies");
      if (!button) {
        button = element("button", "mm-action", "Load more movies");
        button.id = "mm-more-movies";
        button.addEventListener("click", function () {
          if (state.moviesPage < state.moviesTotalPages) {
            state.moviesPage += 1;
            loadMoviesPage();
          } else {
            button.disabled = true;
            button.textContent = "All movies loaded";
          }
        });
        container.appendChild(button);
      }
      if (state.moviesPage >= state.moviesTotalPages) {
        button.disabled = true;
        button.textContent = "All movies loaded";
      }
    } catch (error) {
      stateMessage(container, "Could not load movies: " + error.message);
    }
  }

  async function loadSeriesPage() {
    var container = document.getElementById("page-tv");
    if (!container) return;
    try {
      var response = await apiGet("/series?page=" + state.seriesPage + "&per_page=" + PAGE_SIZE + "&sort=popularity_desc");
      if (state.seriesPage === 1) {
        sectionHeading(container, "More web series", response.meta.total + " titles available");
      }
      if (!response.data.length) stateMessage(container, "No more series to load.");
      else grid(container, response.data);
      state.seriesTotalPages = response.meta.total_pages || 1;
      var button = document.getElementById("mm-more-series");
      if (!button) {
        button = element("button", "mm-action", "Load more web series");
        button.id = "mm-more-series";
        button.addEventListener("click", function () {
          if (state.seriesPage < state.seriesTotalPages) {
            state.seriesPage += 1;
            loadSeriesPage();
          } else {
            button.disabled = true;
            button.textContent = "All series loaded";
          }
        });
        container.appendChild(button);
      }
      if (state.seriesPage >= state.seriesTotalPages) {
        button.disabled = true;
        button.textContent = "All series loaded";
      }
    } catch (error) {
      stateMessage(container, "Could not load web series: " + error.message);
    }
  }

  async function renderMyList() {
    var container = document.getElementById("page-mylist");
    if (!container) return;
    var dynamic = container.querySelector("#mm-my-list-dynamic");
    if (!dynamic) {
      dynamic = element("div");
      dynamic.id = "mm-my-list-dynamic";
      container.appendChild(dynamic);
    }
    dynamic.replaceChildren();
    if (!state.myList.length) {
      dynamic.appendChild(element("p", "mm-message", "Use the + button on any title to add it here."));
      return;
    }
    sectionHeading(dynamic, "Saved titles", state.myList.length + " saved");
    var results = await Promise.all(state.myList.map(function (id) {
      return apiGet("/titles/" + encodeURIComponent(id)).then(function (response) {
        return response.data;
      }).catch(function () { return null; });
    }));
    var valid = results.filter(Boolean);
    if (!valid.length) stateMessage(dynamic, "Saved titles could not be loaded.");
    else grid(dynamic, valid);
  }

  function renderShuffle() {
    var container = document.getElementById("page-shuffle");
    if (!container) return;
    var dynamic = container.querySelector("#mm-shuffle-dynamic");
    if (!dynamic) {
      dynamic = element("div");
      dynamic.id = "mm-shuffle-dynamic";
      container.appendChild(dynamic);
    }
    dynamic.replaceChildren();
    var button = element("button", "mm-action", "Pick something for me");
    button.addEventListener("click", function () {
      var pool = state.trending.length ? state.trending : state.shufflePool || [];
      if (!pool.length) return;
      var picked = pool[Math.floor(Math.random() * pool.length)];
      openDetails(picked.content_id);
    });
    dynamic.appendChild(button);
    dynamic.appendChild(element("p", "mm-note", "Picks from the live trending catalogue."));
  }

  async function runSearch(query) {
    var container = document.getElementById("page-search");
    if (!container) return;
    var dynamic = container.querySelector("#mm-search-dynamic");
    if (!dynamic) {
      dynamic = element("div");
      dynamic.id = "mm-search-dynamic";
      container.appendChild(dynamic);
    }
    if (!query) {
      dynamic.replaceChildren();
      return;
    }
    dynamic.replaceChildren(element("p", "mm-note", "Searching…"));
    try {
      var response = await apiGet("/search?q=" + encodeURIComponent(query) + "&limit=50");
      dynamic.replaceChildren();
      sectionHeading(dynamic, "Results for “" + query + "”", response.data.length + " found");
      if (!response.data.length) stateMessage(dynamic, "No movies or series matched that search.");
      else grid(dynamic, response.data);
    } catch (error) {
      dynamic.replaceChildren(element("p", "mm-note", "Search failed: " + error.message));
    }
  }

  function bindStaticCards() {
    document.querySelectorAll(".card img[src]").forEach(function (image) {
      var filename = image.getAttribute("src").split("/").pop().split("?")[0];
      var contentId = STATIC_ID_MAP[filename];
      if (!contentId) {
        var numeric = filename.match(/^(\d+)\.(?:jpg|jpeg|png|webp)$/i);
        if (numeric) contentId = "movie-" + numeric[1];
      }
      if (!contentId) return;
      var card = image.closest(".card");
      if (!card || card.dataset.mmBound) return;
      card.dataset.mmBound = "true";
      card.style.cursor = "pointer";
      card.addEventListener("click", function () { openDetails(contentId); });
    });
  }

  function bindNavigation() {
    document.querySelectorAll(".nav-item").forEach(function (item) {
      item.setAttribute("role", "button");
      // A <div> defaults to tabIndex -1; -1 is truthy, so check the value.
      if (item.tabIndex < 0) item.tabIndex = 0;
      if (!item.getAttribute("aria-label") && item.title) {
        item.setAttribute("aria-label", item.title);
      }
      item.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          item.click();
        }
      });
      item.addEventListener("click", function () {
        document.querySelectorAll(".nav-item").forEach(function (nav) {
          nav.removeAttribute("aria-current");
        });
        item.setAttribute("aria-current", "page");
        var target = item.getAttribute("data-target");
        if (target === "page-mylist") renderMyList();
        if (target === "page-shuffle") renderShuffle();
      });
    });
    var search = document.querySelector('#page-search .search-input');
    if (search) {
      search.setAttribute("type", "search");
      search.setAttribute("aria-label", "Search movies and series");
      search.addEventListener("input", function () {
        clearTimeout(state.searchTimer);
        var query = search.value.trim();
        state.searchTimer = setTimeout(function () { runSearch(query); }, 320);
      });
      search.addEventListener("keydown", function (event) {
        if (event.key === "Enter") event.preventDefault();
      });
    }
    window.addEventListener("popstate", function (event) {
      if (event.state && event.state.mmTitle) openDetails(event.state.mmTitle, false);
      else closeDetails();
    });
  }

  async function init() {
    injectStyles();
    bindStaticCards();
    bindNavigation();
    var activeNav = document.querySelector(".nav-item.active");
    if (activeNav) activeNav.setAttribute("aria-current", "page");
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeDetails();
    });
    renderMyList();
    renderShuffle();

    var movieTrending = loadSection(
      "page-trending", "/trending?limit=8&media_type=movie", "Trending movies", "live from TMDB catalogue"
    );
    var seriesTrending = apiGet("/trending?limit=8&media_type=tv").then(function (response) {
      return response.data;
    }).catch(function () { return []; });

    Promise.all([movieTrending, seriesTrending]).then(function (results) {
      state.trending = (results[0] || []).concat(results[1] || []);
      state.shufflePool = state.trending;
      if (results[1] && results[1].length) {
        var trendingContainer = document.getElementById("page-trending");
        if (trendingContainer && !document.getElementById("mm-trending-series")) {
          sectionHeading(trendingContainer, "Trending series", "live from the catalogue");
          var seriesGrid = grid(trendingContainer, results[1], "top pick");
          seriesGrid.id = "mm-trending-series";
        }
      }
      renderShuffle();
    });

    loadMoviesPage();
    loadSeriesPage();

    if (config.initialTitle) openDetails(config.initialTitle, false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
