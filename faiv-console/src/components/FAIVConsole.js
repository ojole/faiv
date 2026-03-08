import React, { useState, useEffect, useRef, useCallback } from "react";
import "./FAIVConsole.css";

/****************************************
 * 0) API Base URL (configurable via env)
 ****************************************/
const API_BASE =
  process.env.REACT_APP_API_BASE_URL || "https://api.faiv.ai";
const API_RETRY_ATTEMPTS = 4;
const API_RETRY_DELAY_MS = 400;
const API_FETCH_TIMEOUT_MS = 45000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function isNetworkFetchError(err) {
  if (!err) return false;
  const message = String(err.message || "");
  return (
    err.name === "AbortError" ||
    err.name === "TypeError" ||
    message === "Failed to fetch" ||
    message === "Load failed" ||
    /NetworkError/i.test(message)
  );
}

function makeRequestId(prefix = "req") {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function getStorageItem(storageObj, key, fallback = "") {
  try {
    return storageObj?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function setStorageItem(storageObj, key, value) {
  try {
    storageObj?.setItem(key, value);
  } catch {
    // Some embedded/private browser contexts deny storage writes.
  }
}

async function fetchWithRetry(url, options = {}, attempts = API_RETRY_ATTEMPTS) {
  const { timeoutMs = API_FETCH_TIMEOUT_MS, ...fetchOptions } = options;
  let lastErr = null;
  for (let attempt = 0; attempt <= attempts; attempt += 1) {
    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timeoutId = controller
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;

    try {
      return await fetch(url, {
        ...fetchOptions,
        signal: controller ? controller.signal : undefined,
      });
    } catch (err) {
      lastErr = err;
      if (!isNetworkFetchError(err) || attempt >= attempts) {
        throw err;
      }
      const jitter = Math.floor(Math.random() * 200);
      await sleep(API_RETRY_DELAY_MS * (attempt + 1) + jitter);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }
  throw lastErr || new Error("Unknown network error");
}

/****************************************
 * 1) ASCII Loader Frames
 ****************************************/
const asciiFAIVFrames = [
  [
    "███████╗ █████╗ ██╗██╗   ██╗",
    "██╔════╝██╔══██╗██║██║   ██║",
    "█████╗  ███████║██║██║   ██║",
    "██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
    "██║     ██║  ██║██║ ╚████╔╝ ",
    "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ",
  ],
  [
    " ██████╗ █████╗ ██╗██╗   ██╗",
    "██╔════╝██╔══██╗██║██║   ██║",
    "█████╗  ███████║██║██║   ██║",
    "██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
    "██║     ██║  ██║██║ ╚████╔╝",
    "╚═╝     ██╔═╝  ╚═╝╚═╝  ╚═══╝ ",
  ],
  [
    "  ██████╗ █████╗ ██╗██╗   ██╗",
    " ██╔════╝██╔══██╗██║██║   ██║",
    " █████╗  ███████║██║██║   ██║",
    " ██╔══╝  ██╔══██║██║╚██╗ ██╔╝",
    " ██║     ██║  ██║██║ ╚████╔╝ ",
    " ╚═╝     ██╔═╝  ╚═╝╚═╝  ╚═══╝  ",
  ],
];
// New "V" logo frames
const asciiVFrames = [
  [
    "██╗   ██╗",
    "██║   ██║",
    "██║   ██║",
    "╚██╗ ██╔╝",
    " ╚████╔╝ ",
    "  ╚═══╝  ",
  ],
  [
    " ██╗   ██╗",
    "██║   ██║",
    "██║   ██║",
    "╚██╗ ██╔╝",
    " ╚████╔╝",
    "  ╚═══╝ ",
  ],
  [
    "  ██╗   ██╗",
    " ██║   ██║",
    " ██║   ██║",
    " ╚██╗ ██╔╝",
    "  ╚████╔╝ ",
    "   ╚═══╝  ",
  ],
];

/****************************************
 * 2) Helper to parse final FAIV output
 ****************************************/
function extractFinalOutput(response) {
  if (!response || typeof response !== "string") {
    return "No valid response received.";
  }

  // 1) Clean up zero-width etc.
  const cleaned = response
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/\s*\n\s*/g, "\n")
    .trim();

  // 2) Split by newlines
  const lines = cleaned.split("\n");

  // 3) Minor helper to remove stray bracket/asterisk combos
  const cleanStr = (str) => str.replace(/\]*:?[*]+/g, "").trim();

  // 4) Return a <div> of parsed lines
  return (
    <div className="output-block">
      {lines.map((rawLine, idx) => {
        const line = rawLine.trim();

        // CASE A: "Consensus:"
        if (/Consensus:/i.test(line)) {
          const [labelPart, afterLabel] = line.split(/Consensus:\s*/i);
          const labelClean = cleanStr(labelPart + "Consensus:");
          const content = cleanStr(afterLabel || "");
          return (
            <div key={idx} className="console-line">
              <b>
                <u>{labelClean}</u>
              </b>
              {": "}
              {content}
            </div>
          );
        }

        // CASE B: "Confidence Score:"
        if (line.startsWith("Confidence Score:")) {
          const val = cleanStr(line.replace(/^Confidence Score:/i, ""));
          return (
            <div key={idx} className="console-line">
              <b>
                <u>Confidence Score:</u>
              </b>{" "}
              {val}
            </div>
          );
        }

        // CASE C: "Justification:"
        if (line.startsWith("Justification:")) {
          const val = cleanStr(line.replace(/^Justification:/i, ""));
          return (
            <div key={idx} className="console-line">
              <b>
                <u>Justification:</u>
              </b>{" "}
              {val}
            </div>
          );
        }

        // CASE D: "Differing Opinion -"
        if (line.startsWith("Differing Opinion -")) {
          const val = cleanStr(line.replace(/^Differing Opinion -/i, ""));
          return (
            <div key={idx} className="console-line">
              <b>
                <u>Differing Opinion -</u>
              </b>{" "}
              {val}
            </div>
          );
        }

        // CASE E: "Reason:"
        if (line.startsWith("Reason:")) {
          const val = cleanStr(line.replace(/^Reason:/i, ""));
          return (
            <div key={idx} className="console-line">
              <b>
                <u>Reason:</u>
              </b>{" "}
              {val}
            </div>
          );
        }

        // Fallback
        return (
          <div key={idx} className="console-line">
            {cleanStr(line)}
          </div>
        );
      })}
    </div>
  );
}

/****************************************
 * 3) Summarize user input for chat title
 ****************************************/
function summarizeInput(text) {
  // Strip trailing punctuation
  let s = text.replace(/[?.!,;:]+$/g, "").trim();
  // Remove leading filler words (question starters, articles, etc.)
  const fillerPattern = /^(what|which|where|when|who|how|why|can|could|would|should|do|does|is|are|was|were|tell|me|about|give|please|i\s+want\s+to\s+know|i\s+need|i\s+want|the|a|an|some|any|be\s+considered|considered|regarded\s+as|thought\s+of\s+as|known\s+as|i)\s+/i;
  let prev = "";
  while (s !== prev) {
    prev = s;
    s = s.replace(fillerPattern, "").trim();
  }
  // Lowercase the result
  s = s.toLowerCase();
  // Cap at 40 chars on a word boundary
  if (s.length > 40) {
    s = s.slice(0, 40).replace(/\s+\S*$/, "").trim();
  }
  // Fallback if we stripped everything
  if (!s) {
    s = text.slice(0, 30).trim().toLowerCase();
  }
  return `"${s}"`;
}

/****************************************
 * 4) Parse deliberation into speaker entries
 ****************************************/
function parseDeliberation(delibText) {
  if (!delibText) return [];
  const lines = delibText.split("\n");
  const entries = [];
  const speakerRegex = /^<?(\w+)>?\s*\(([^)]+)\)\s*:\s*(.+)/;
  let currentEntry = null;

  for (const line of lines) {
    const match = line.match(speakerRegex);
    if (match) {
      if (currentEntry) entries.push(currentEntry);
      currentEntry = {
        member: match[1],
        pillar: match[2].trim(),
        text: match[3].trim(),
      };
    } else if (currentEntry && line.trim()) {
      currentEntry.text += " " + line.trim();
    }
  }
  if (currentEntry) entries.push(currentEntry);
  return entries;
}

/****************************************
 * 5) Deliberation Tile (single speaker)
 ****************************************/
function DeliberationTile({ entry, index, onSubmitComment, isSubmitting, replyMsgIdx, onJumpToReply, tileId }) {
  const [expanded, setExpanded] = React.useState(false);
  const [comment, setComment] = React.useState("");

  function handleToggle() {
    setExpanded((prev) => !prev);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!comment.trim()) return;
    onSubmitComment(entry, index, comment);
    setComment("");
    setExpanded(false);
  }

  return (
    <div className="deliberation-tile" id={tileId} onClick={handleToggle}>
      <div className="tile-header">
        <span className="tile-member">{entry.member}</span>
        <span className="tile-pillar">({entry.pillar})</span>
        {replyMsgIdx !== undefined && (
          <span
            className="tile-reply-badge"
            title="Jump to your reply"
            onClick={(e) => {
              e.stopPropagation();
              onJumpToReply(replyMsgIdx);
            }}
          >
            replied
          </span>
        )}
      </div>
      <div className="tile-text">{entry.text}</div>
      {expanded && (
        <form className="tile-comment-form" onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
          <input
            className="tile-comment-input"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Agree or disagree..."
            autoFocus
          />
          <button
            type="submit"
            className="tile-comment-submit"
            disabled={isSubmitting}
          >
            {isSubmitting ? "..." : "Reply"}
          </button>
        </form>
      )}
    </div>
  );
}

/****************************************
 * 6) Deliberation Panel (all tiles)
 ****************************************/
function DeliberationPanel({ delibText, onSubmitComment, isSubmitting, repliedTiles, delibKey, onJumpToReply }) {
  const entries = parseDeliberation(delibText);

  if (entries.length === 0) {
    return <pre className="deliberation-pre">{delibText || "Deliberation unavailable."}</pre>;
  }

  return (
    <div className="deliberation-tiles-container">
      {entries.map((entry, idx) => {
        const tileKey = `${delibKey}-${idx}`;
        // Skip tiles that have been replied to — they're rendered outside the drawer
        if (repliedTiles?.[tileKey] !== undefined) return null;
        return (
          <DeliberationTile
            key={idx}
            entry={entry}
            index={idx}
            tileId={`tile-${tileKey}`}
            onSubmitComment={onSubmitComment}
            isSubmitting={isSubmitting}
            onJumpToReply={onJumpToReply}
          />
        );
      })}
    </div>
  );
}

/* Helper: extract replied-to tiles from a deliberation for rendering outside the drawer */
function getRepliedTileEntries(delibText, delibKey, repliedTiles) {
  if (!delibText || !repliedTiles) return [];
  const entries = parseDeliberation(delibText);
  const result = [];
  entries.forEach((entry, idx) => {
    const tileKey = `${delibKey}-${idx}`;
    if (repliedTiles[tileKey] !== undefined) {
      result.push({ entry, idx, tileKey, replyMsgIdx: repliedTiles[tileKey] });
    }
  });
  return result;
}

/****************************************
 * 7) Main Component
 ****************************************/
export default function FAIVConsole() {
  // All sessions + selected session ID
  const [allSessions, setAllSessions] = useState({});
  const [activeSessionId, setActiveSessionId] = useState("");

  // Input + Loading
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // ASCII wave frames
  const [asciiFrame, setAsciiFrame] = useState(0);
  const [progress, setProgress] = useState(0);

  const [useVLogo, setUseVLogo] = useState(false);

  // Pillar dropdown
  const [selectedPillar, setSelectedPillar] = useState("FAIV");
  const [pillarOpen, setPillarOpen] = useState(false);

  // Delete confirm modal
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [pendingDeleteSession, setPendingDeleteSession] = useState("");
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");

  // API health status
  const [apiStatus, setApiStatus] = useState("checking"); // "ok" | "down" | "checking"
  const [authStatusLoading, setAuthStatusLoading] = useState(true);
  const [passwordProtected, setPasswordProtected] = useState(false);
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [unlockPassword, setUnlockPassword] = useState("");
  const [unlockSubmitting, setUnlockSubmitting] = useState(false);
  const [unlockError, setUnlockError] = useState("");
  const [lockAsciiFrame, setLockAsciiFrame] = useState(0);
  const [embedSessionToken, setEmbedSessionToken] = useState(() => {
    try {
      if (typeof window === "undefined") return "";
      const params = new URLSearchParams(window.location.search);
      return params.get("embed_session") || getStorageItem(sessionStorage, "faiv_embed_session", "");
    } catch {
      return "";
    }
  });

  // Store deliberation text per message index (persisted in localStorage)
  const [deliberations, setDeliberations] = useState(() => {
    try {
      const stored = getStorageItem(localStorage, "faiv_deliberations", "");
      return stored ? JSON.parse(stored) : {};
    } catch { return {}; }
  });

  // Re-deliberation loading state
  const [redeliberating, setRedeliberating] = useState(false);

  // Track which tiles have replies: { "delibKey-entryIdx": replyMsgIdx } (persisted in localStorage)
  const [repliedTiles, setRepliedTiles] = useState(() => {
    try {
      const stored = getStorageItem(localStorage, "faiv_replied_tiles", "");
      return stored ? JSON.parse(stored) : {};
    } catch { return {}; }
  });

  // Error details (for expandable display)
  const [lastError, setLastError] = useState(null);

  // Mobile history panel toggle
  const [historyOpen, setHistoryOpen] = useState(false);

  const consoleBodyRef = useRef(null);

  function scrollToElement(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const withEmbedSessionHeader = useCallback((headers = {}) => {
    if (!embedSessionToken) return headers;
    return {
      ...headers,
      "X-FAIV-Embed-Session": embedSessionToken,
    };
  }, [embedSessionToken]);

  const requestOptions = useCallback(
    (options = {}, forceCookieAuth = false) => {
      const { credentials, ...rest } = options;
      return {
        mode: "cors",
        cache: "no-store",
        ...rest,
        credentials:
          credentials || (forceCookieAuth ? "include" : embedSessionToken ? "omit" : "include"),
      };
    },
    [embedSessionToken]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const embedSessionFromQuery = params.get("embed_session");

    if (embedSessionFromQuery) {
      setEmbedSessionToken(embedSessionFromQuery);
      setStorageItem(sessionStorage, "faiv_embed_session", embedSessionFromQuery);
      const canPersist = getStorageItem(sessionStorage, "faiv_embed_session", "") === embedSessionFromQuery;

      if (canPersist) {
        params.delete("embed_session");
        const nextQuery = params.toString();
        const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}${window.location.hash}`;
        window.history.replaceState({}, document.title, nextUrl);
      }
      return;
    }

    try {
      const storedSession = getStorageItem(sessionStorage, "faiv_embed_session", "");
      if (storedSession && storedSession !== embedSessionToken) {
        setEmbedSessionToken(storedSession);
      }
    } catch {
      // Ignore storage read failures in restricted browsing contexts.
    }
  }, [embedSessionToken]);

  useEffect(() => {
    let alive = true;

    async function checkAuthStatus() {
      try {
        const resp = await fetchWithRetry(
          `${API_BASE}/api/auth-status`,
          requestOptions({
            headers: withEmbedSessionHeader(),
          })
        );
        if (!alive) return;
        if (!resp.ok) {
          setPasswordProtected(true);
          setIsUnlocked(false);
          return;
        }
        const data = await resp.json();
        const protectedMode = Boolean(data?.passwordProtected);
        const authenticated = Boolean(data?.authenticated);
        setPasswordProtected(protectedMode);
        setIsUnlocked(!protectedMode || authenticated);
        setUnlockError("");
      } catch {
        if (!alive) return;
        // Security-first: fail closed if auth status cannot be verified.
        setPasswordProtected(true);
        setIsUnlocked(false);
        setUnlockError("Unable to verify access status. Try again.");
      } finally {
        if (alive) setAuthStatusLoading(false);
      }
    }

    checkAuthStatus();
    return () => {
      alive = false;
    };
  }, [requestOptions, withEmbedSessionHeader]);

  // Health check on mount + periodic
  useEffect(() => {
    async function checkHealth() {
      try {
        const resp = await fetchWithRetry(
          `${API_BASE}/health`,
          requestOptions({
            headers: withEmbedSessionHeader(),
          })
        );
        if (resp.ok) {
          setApiStatus("ok");
        } else {
          setApiStatus("down");
        }
      } catch {
        setApiStatus("down");
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [requestOptions, withEmbedSessionHeader]);

  async function handleUnlockSubmit(e) {
    e.preventDefault();
    if (!unlockPassword.trim()) return;
    setUnlockSubmitting(true);
    setUnlockError("");
    try {
      const resp = await fetchWithRetry(
        `${API_BASE}/api/unlock`,
        requestOptions(
          {
            method: "POST",
            headers: withEmbedSessionHeader({ "Content-Type": "application/json", Accept: "application/json" }),
            body: JSON.stringify({ password: unlockPassword }),
          },
          true
        )
      );

      if (!resp.ok) {
        const payload = await resp.json().catch(() => ({}));
        setUnlockError(payload.error || "Unlock failed.");
        return;
      }

      setIsUnlocked(true);
      setUnlockPassword("");
      setApiStatus("ok");
    } catch {
      setUnlockError("Unable to reach unlock endpoint.");
    } finally {
      setUnlockSubmitting(false);
    }
  }

  useEffect(() => {
    if (!(passwordProtected && !isUnlocked)) {
      setLockAsciiFrame(0);
      return;
    }
    const interval = setInterval(() => {
      setLockAsciiFrame((prev) => (prev + 1) % asciiFAIVFrames.length);
    }, 320);
    return () => clearInterval(interval);
  }, [passwordProtected, isUnlocked]);

  // 2) On mount => load existing sessions from localStorage
  useEffect(() => {
    const stored = getStorageItem(localStorage, "faiv_sessions", "");
    if (stored) {
      const parsed = JSON.parse(stored);
      setAllSessions(parsed);

      const lastActive = getStorageItem(localStorage, "faiv_session_id", "");
      if (lastActive && parsed[lastActive]) {
        setActiveSessionId(lastActive);
      } else {
        const keys = Object.keys(parsed);
        if (keys.length > 0) {
          setActiveSessionId(keys[0]);
          setStorageItem(localStorage, "faiv_session_id", keys[0]);
        } else {
          handleNewChat();
        }
      }
    } else {
      handleNewChat();
    }
  }, []);

  // 3) Whenever sessions change => persist
  useEffect(() => {
    setStorageItem(localStorage, "faiv_sessions", JSON.stringify(allSessions));
  }, [allSessions]);

  // 3b) Persist deliberations and replied tiles
  useEffect(() => {
    setStorageItem(localStorage, "faiv_deliberations", JSON.stringify(deliberations));
  }, [deliberations]);

  useEffect(() => {
    setStorageItem(localStorage, "faiv_replied_tiles", JSON.stringify(repliedTiles));
  }, [repliedTiles]);

  // 4) Animate progress bar
  useEffect(() => {
    let interval;
    if (loading) {
      setProgress(0);
      const startTime = Date.now();
      interval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const newVal = Math.min((elapsed / 8000) * 100, 99);
        setProgress(newVal);
      }, 500);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // 5) Animate ASCII frames
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setAsciiFrame((prev) => {
          const frames = useVLogo ? asciiVFrames : asciiFAIVFrames;
          return (prev + 1) % frames.length;
        });
      }, 400);
    } else {
      setAsciiFrame(0);
      setUseVLogo((prev) => !prev);
    }
    return () => clearInterval(interval);
  }, [loading, useVLogo]);

  // 6) Helpers
  const currentSession = allSessions[activeSessionId];
  const currentMessages = currentSession ? currentSession.messages : [];

  function updateSessionMessages(sessionId, newMessages) {
    setAllSessions((prev) => ({
      ...prev,
      [sessionId]: {
        ...prev[sessionId],
        messages: newMessages,
      },
    }));
  }

  function handleNewChat() {
    const newId = crypto.randomUUID();
    const newTitle = "Untitled";
    const newSession = { title: newTitle, messages: [] };
    setAllSessions((prev) => ({ ...prev, [newId]: newSession }));
    setActiveSessionId(newId);
    setStorageItem(localStorage, "faiv_session_id", newId);
    setHistoryOpen(false);
  }

  function ensureAtLeastOneSession() {
    const keys = Object.keys(allSessions);
    if (keys.length === 0) {
      handleNewChat();
    }
  }

  function handleDeleteChat(sessionId) {
    setPendingDeleteSession(sessionId);
    setDeleteConfirmInput("");
    setShowDeleteModal(true);
  }

  function confirmDelete() {
    if (deleteConfirmInput.toLowerCase().trim() !== "delete") return;
    setShowDeleteModal(false);

    const sessId = pendingDeleteSession;
    setAllSessions((prev) => {
      const copy = { ...prev };
      delete copy[sessId];
      return copy;
    });

    // Clean up deliberations and replied tiles for deleted session
    setDeliberations((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((key) => {
        if (key.startsWith(sessId + "-")) delete copy[key];
      });
      return copy;
    });
    setRepliedTiles((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((key) => {
        if (key.startsWith(sessId + "-")) delete copy[key];
      });
      return copy;
    });

    if (sessId === activeSessionId) {
      const remain = Object.keys(allSessions).filter((id) => id !== sessId);
      if (remain.length > 0) {
        setActiveSessionId(remain[0]);
        setStorageItem(localStorage, "faiv_session_id", remain[0]);
      } else {
        handleNewChat();
      }
    }
  }

  function handleSelectSession(sessId) {
    setActiveSessionId(sessId);
    setStorageItem(localStorage, "faiv_session_id", sessId);
    setHistoryOpen(false);
  }

  // Submit input => call server
  async function handleSubmit(e) {
    e.preventDefault();

    ensureAtLeastOneSession();
    if (!input.trim() || !activeSessionId) return;

    if (!allSessions[activeSessionId]) {
      handleNewChat();
      return;
    }

    const title = allSessions[activeSessionId].title;
    let updated = [...allSessions[activeSessionId].messages];

    if (updated.length > 0) updated.push("-----");
    updated.push(`> ${input}`);

    if (title === "Untitled") {
      let snippet = summarizeInput(input);
      setAllSessions((prev) => ({
        ...prev,
        [activeSessionId]: { ...prev[activeSessionId], title: snippet },
      }));
    }

    const capturedInput = input;
    updateSessionMessages(activeSessionId, updated);
    setInput("");
    setLoading(true);
    setLastError(null);
    const requestId = makeRequestId("query");

    try {
      const resp = await fetchWithRetry(
        `${API_BASE}/query/`,
        requestOptions({
          method: "POST",
          headers: withEmbedSessionHeader({
            "Content-Type": "application/json",
            Accept: "application/json",
          }),
          body: JSON.stringify({
            session_id: activeSessionId,
            input_text: capturedInput,
            pillar: selectedPillar,
            request_id: requestId,
          }),
        })
      );
      if (!resp.ok) {
        const errorBody = await resp.text();
        if (resp.status === 401) {
          setPasswordProtected(true);
          setIsUnlocked(false);
          setUnlockError("Access session expired. Please unlock again.");
        }
        throw new Error(`HTTP ${resp.status}: ${errorBody}`);
      }
      const data = await resp.json();

      updated.push(data.response);
      updateSessionMessages(activeSessionId, updated);

      // Store deliberation with metadata for re-deliberation
      if (data.deliberation) {
        const msgIdx = updated.length - 1;
        setDeliberations((prev) => ({
          ...prev,
          [`${activeSessionId}-${msgIdx}`]: {
            deliberation: data.deliberation,
            originalInput: capturedInput,
            council: data.council || {},
          },
        }));
      }

      // Refresh health status on success
      setApiStatus("ok");
    } catch (err) {
      const isNetworkError = isNetworkFetchError(err);
      const displayMsg = isNetworkError
        ? "Cannot reach FAIV API right now. Please retry."
        : `API error: ${err.message}`;
      updated.push(`[ERROR] ${displayMsg}`);
      updateSessionMessages(activeSessionId, updated);
      setLastError(`${err.name || "Error"}: ${err.message}`);
      if (isNetworkError) {
        setApiStatus("down");
      }
    } finally {
      setLoading(false);
    }

    setTimeout(() => {
      if (consoleBodyRef.current) {
        consoleBodyRef.current.scrollTop = consoleBodyRef.current.scrollHeight;
      }
    }, 100);
  }

  async function handleRedeliberate(entry, entryIndex, comment, msgIdx) {
    const delibKey = `${activeSessionId}-${msgIdx}`;
    const delibData = deliberations[delibKey];
    if (!delibData) return;

    const delibText = typeof delibData === "string" ? delibData : delibData.deliberation;
    const originalInput = typeof delibData === "string" ? "" : delibData.originalInput;
    const councilNames = typeof delibData === "string" ? [] : Object.keys(delibData.council || {});

    // Reconstruct deliberation up to and including the clicked tile
    const entries = parseDeliberation(delibText);
    const entriesUpTo = entries.slice(0, entryIndex + 1);
    const deliberationUpTo = entriesUpTo
      .map((e) => `${e.member} (${e.pillar}): ${e.text}`)
      .join("\n");

    setRedeliberating(true);
    setLoading(true);
    setLastError(null);

    let updated = [...(allSessions[activeSessionId]?.messages || [])];
    updated.push("-----");
    const replyMsg = `└ [Reply to ${entry.member}]: ${comment}`;
    updated.push(replyMsg);
    const replyMsgIdx = updated.length - 1;
    updateSessionMessages(activeSessionId, updated);

    // Track the tile-to-reply link
    const tileKey = `${activeSessionId}-${msgIdx}-${entryIndex}`;
    setRepliedTiles((prev) => ({ ...prev, [tileKey]: replyMsgIdx }));
    const requestId = makeRequestId("redeliberate");

    try {
      const resp = await fetchWithRetry(
        `${API_BASE}/redeliberate/`,
        requestOptions({
          method: "POST",
          headers: withEmbedSessionHeader({ "Content-Type": "application/json", Accept: "application/json" }),
          body: JSON.stringify({
            session_id: activeSessionId,
            original_input: originalInput,
            deliberation_up_to: deliberationUpTo,
            user_comment: comment,
            target_member: entry.member,
            pillar: selectedPillar,
            council_members: councilNames,
            request_id: requestId,
          }),
        })
      );
      if (!resp.ok) {
        const errorBody = await resp.text();
        if (resp.status === 401) {
          setPasswordProtected(true);
          setIsUnlocked(false);
          setUnlockError("Access session expired. Please unlock again.");
        }
        throw new Error(`HTTP ${resp.status}: ${errorBody}`);
      }
      const data = await resp.json();

      updated.push(data.response);
      updateSessionMessages(activeSessionId, updated);

      if (data.deliberation) {
        const newMsgIdx = updated.length - 1;
        setDeliberations((prev) => ({
          ...prev,
          [`${activeSessionId}-${newMsgIdx}`]: {
            deliberation: data.deliberation,
            originalInput: originalInput,
            council: data.council || {},
          },
        }));
      }
      setApiStatus("ok");
    } catch (err) {
      const isNetworkError = isNetworkFetchError(err);
      const displayMsg = isNetworkError
        ? "Cannot reach FAIV API right now. Please retry."
        : `API error: ${err.message}`;
      updated.push(`[ERROR] ${displayMsg}`);
      updateSessionMessages(activeSessionId, updated);
      setLastError(`${err.name || "Error"}: ${err.message}`);
      if (isNetworkError) setApiStatus("down");
    } finally {
      setLoading(false);
      setRedeliberating(false);
    }

    setTimeout(() => {
      if (consoleBodyRef.current) {
        consoleBodyRef.current.scrollTop = consoleBodyRef.current.scrollHeight;
      }
    }, 100);
  }

  async function handleResetSession() {
    if (!activeSessionId) return;
    try {
      await fetchWithRetry(
        `${API_BASE}/reset/?session_id=${encodeURIComponent(activeSessionId)}`,
        requestOptions({
          method: "POST",
          headers: withEmbedSessionHeader(),
        })
      );
    } catch {
      // Backend reset is best-effort; clear locally regardless
    }
    // Clear local session messages
    setAllSessions((prev) => ({
      ...prev,
      [activeSessionId]: { ...prev[activeSessionId], messages: [], title: "Untitled" },
    }));
    // Clean up deliberations and replied tiles for this session only
    setDeliberations((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((key) => {
        if (key.startsWith(activeSessionId + "-")) delete copy[key];
      });
      return copy;
    });
    setRepliedTiles((prev) => {
      const copy = { ...prev };
      Object.keys(copy).forEach((key) => {
        if (key.startsWith(activeSessionId + "-")) delete copy[key];
      });
      return copy;
    });
    setLastError(null);
  }

  /****************************************
   * RENDER
   ****************************************/
  if (authStatusLoading) {
    return (
      <div className="outer-container lock-screen-shell">
        <div className="retro-window lock-window">
          <div className="retro-title-bar lock-title-bar">PASSWORD PROTECTED</div>
          <div className="lock-window-body">Initializing access control...</div>
        </div>
      </div>
    );
  }

  if (passwordProtected && !isUnlocked) {
    return (
      <div className="outer-container lock-screen-shell">
        <div className="lock-stack">
          <div className="lock-ascii-wrap" aria-hidden="true">
            {asciiFAIVFrames[lockAsciiFrame].map((line, i) => (
              <pre key={i} className="lock-ascii-line">
                {line}
              </pre>
            ))}
          </div>
          <div className="retro-window lock-window">
            <div className="retro-title-bar lock-title-bar">PASSWORD PROTECTED</div>
            <div className="lock-window-body">
              <form className="lock-form" onSubmit={handleUnlockSubmit}>
                <input
                  type="password"
                  className="lock-input"
                  value={unlockPassword}
                  onChange={(event) => setUnlockPassword(event.target.value)}
                  placeholder="Enter access password..."
                  autoFocus
                />
                <button type="submit" className="lock-submit" disabled={unlockSubmitting}>
                  {unlockSubmitting ? "UNLOCKING..." : "UNLOCK"}
                </button>
              </form>
              <div className="lock-error">{unlockError || "\u00A0"}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="outer-container">
      <div className="windows-container">
        {/* Mobile backdrop */}
        <div
          className={`history-backdrop ${historyOpen ? "visible" : ""}`}
          onClick={() => setHistoryOpen(false)}
        />

        {/* LEFT WINDOW */}
        <div className={`retro-window left-window ${historyOpen ? "mobile-open" : ""}`}>
          <div className="retro-title-bar left-title-bar">
            <span>History</span>
            <button
              className="new-chat-btn"
              onClick={handleNewChat}
              title="Start New Chat"
            >
              &gt;
            </button>
          </div>

          <div className="left-window-body">
            {Object.keys(allSessions).map((sessId) => {
              const info = allSessions[sessId];
              return (
                <div
                  key={sessId}
                  className={`chat-item ${sessId === activeSessionId ? "active" : ""}`}
                  onClick={() => handleSelectSession(sessId)}
                >
                  <span>{info.title || "Untitled"}</span>
                  <button
                    className="chat-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteChat(sessId);
                    }}
                    title="Delete Chat"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="#00ff00">
                      <path d="M3 6h18M8 6v14c0 .55.45 1 1 1h6c.55 0 1-.45 1-1V6" stroke="none" />
                      <path d="M10 9v8M14 9v8" stroke="black" strokeWidth="2" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* RIGHT WINDOW */}
        <div className="retro-window right-window">
          <div className="retro-title-bar right-title-bar">
            {/* Mobile history toggle */}
            <button
              className="history-toggle-btn"
              onClick={() => setHistoryOpen(!historyOpen)}
              title="Toggle History"
            >
              {historyOpen ? "×" : "☰"}
            </button>

            {/* Pillar dropdown */}
            <div
              className="pillar-dropdown-wrapper"
              onClick={() => setPillarOpen(!pillarOpen)}
            >
              <div className="pillar-dropdown-display">{selectedPillar}</div>
              <div className="pillar-arrow">{pillarOpen ? "\u25B2" : "\u25BC"}</div>
            </div>

            <ul className={`pillar-menu ${pillarOpen ? "open" : ""}`}>
              {["FAIV","Wisdom","Strategy","Expansion","Future","Integrity"].map(opt => (
                <li
                  key={opt}
                  className={opt === selectedPillar ? "selected" : ""}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedPillar(opt);
                    setPillarOpen(false);
                  }}
                >
                  {opt}
                </li>
              ))}
            </ul>

            {/* Page title + reset */}
            <div style={{ marginLeft: "10px", marginRight: "auto", display: "flex", alignItems: "center" }}>
              <span className="page-title">Counsole</span>
              <button
                className="reset-btn"
                onClick={handleResetSession}
                title="Reset current session"
              >
                Reset
              </button>
            </div>

            {/* API status indicator */}
            <div className="api-status-wrapper" title={`API: ${apiStatus}`}>
              <span
                className={`api-status-dot ${apiStatus === "ok" ? "status-ok" : apiStatus === "down" ? "status-down" : "status-checking"}`}
              />
            </div>
          </div>

          {/* The main console area + loader */}
          <div
            className={`right-console-body ${loading ? "loading" : ""}`}
            ref={consoleBodyRef}
          >
            {loading ? (
              <div className="ascii-loader">
                {(useVLogo ? asciiVFrames : asciiFAIVFrames)[asciiFrame].map((line, i) => (
                  <pre key={i} className="ascii-logo">
                    {line}
                  </pre>
                ))}
                <div className="progress-bar">
                  {"["}
                  {"\u2588".repeat(Math.round(progress / 10))}
                  {"\u2592".repeat(10 - Math.round(progress / 10))}
                  {"]"}
                </div>
              </div>
            ) : (
              <div className="console-output">
                {currentMessages.map((msg, idx) => {
                  if (msg === "-----") {
                    return <div key={idx} className="separator-line" />;
                  }

                  // Error messages
                  if (typeof msg === "string" && msg.startsWith("[ERROR]")) {
                    return (
                      <div key={idx} className="console-line error-line">
                        {msg.replace("[ERROR] ", "")}
                        {lastError && (
                          <details className="error-details">
                            <summary>Technical details</summary>
                            <pre>{lastError}</pre>
                          </details>
                        )}
                      </div>
                    );
                  }

                  // Consensus responses
                  if (
                    typeof msg === "string" &&
                    (msg.includes("Consensus:") ||
                     msg.includes("Confidence Score:"))
                  ) {
                    const delibKey = `${activeSessionId}-${idx}`;
                    const delibData = deliberations[delibKey];
                    const delibText = typeof delibData === "string" ? delibData : delibData?.deliberation;
                    const repliedTileEntries = getRepliedTileEntries(delibText, delibKey, repliedTiles);
                    return (
                      <div key={idx} className="console-line">
                        {extractFinalOutput(msg)}
                        {/* Replied-to tiles shown outside the drawer — always visible */}
                        {repliedTileEntries.length > 0 && (
                          <div className="replied-tiles-visible">
                            {repliedTileEntries.map(({ entry, idx: entryIdx, tileKey, replyMsgIdx }) => (
                              <DeliberationTile
                                key={tileKey}
                                entry={entry}
                                index={entryIdx}
                                tileId={`tile-${tileKey}`}
                                onSubmitComment={(e, eIdx, comment) =>
                                  handleRedeliberate(e, eIdx, comment, idx)
                                }
                                isSubmitting={redeliberating}
                                replyMsgIdx={replyMsgIdx}
                                onJumpToReply={(replyIdx) => scrollToElement(`msg-${replyIdx}`)}
                              />
                            ))}
                          </div>
                        )}
                        <details className="deliberation-details">
                          <summary className="deliberation-summary">Deliberation</summary>
                          <DeliberationPanel
                            delibText={delibText}
                            delibKey={delibKey}
                            onSubmitComment={(entry, entryIdx, comment) =>
                              handleRedeliberate(entry, entryIdx, comment, idx)
                            }
                            isSubmitting={redeliberating}
                            repliedTiles={repliedTiles}
                            onJumpToReply={(replyIdx) => scrollToElement(`msg-${replyIdx}`)}
                          />
                        </details>
                      </div>
                    );
                  }

                  // Reply messages: └ [Reply to MemberName]: text
                  if (typeof msg === "string" && (msg.startsWith("└ [Reply to ") || msg.startsWith("|_ [Reply to "))) {
                    const memberMatch = msg.match(/^(?:└|\|_) \[Reply to (\w+)\]: (.+)/);
                    const memberName = memberMatch ? memberMatch[1] : null;
                    // Find the tile to jump back to
                    const sourceTileId = memberName
                      ? Object.keys(repliedTiles).find((k) => repliedTiles[k] === idx)
                      : null;
                    return (
                      <div key={idx} id={`msg-${idx}`} className="console-line reply-line">
                        <span
                          className="reply-prefix"
                          title={sourceTileId ? `Jump to ${memberName}'s statement` : undefined}
                          onClick={sourceTileId ? () => scrollToElement(`tile-${sourceTileId}`) : undefined}
                          style={sourceTileId ? { cursor: "pointer" } : undefined}
                        >
                          └
                        </span>{" "}
                        {memberMatch ? (
                          <>
                            <span className="reply-member">[Reply to {memberName}]:</span>{" "}
                            {memberMatch[2]}
                          </>
                        ) : (
                          msg.slice(2)
                        )}
                      </div>
                    );
                  }

                  return (
                    <div key={idx} id={`msg-${idx}`} className="console-line">
                      {msg}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Input form */}
          <form className="console-input" onSubmit={handleSubmit}>
            <input
              className="input-field"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="FAIV awaits your query..."
            />
            <button type="submit" className="submit-btn">
              Enter
            </button>
          </form>
        </div>
      </div>

      {/* Delete Modal */}
      {showDeleteModal && (
        <div className="modal-backdrop">
          <div className="modal-box">
            <h3>Confirm Deletion</h3>
            <p>This will permanently delete the selected chat.</p>
            <p>
              Type <b>"delete"</b> to confirm:
            </p>
            <input
              type="text"
              value={deleteConfirmInput}
              onChange={(e) => setDeleteConfirmInput(e.target.value)}
            />
            <div className="modal-buttons">
              <button
                onClick={confirmDelete}
                disabled={deleteConfirmInput.toLowerCase().trim() !== "delete"}
              >
                Delete
              </button>
              <button onClick={() => setShowDeleteModal(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
