"use strict";

const { invoke, listen, platform } = window.gscpBridge;

// ── DOM elements ──
const el = {
  modeWifi: document.getElementById("mode-wifi"),
  modeUsb: document.getElementById("mode-usb"),
  connectionMode: document.getElementById("connection-mode"),
  wifiSettings: document.getElementById("wifi-settings"),
  usbModeHint: document.getElementById("usb-mode-hint"),
  buttonRow: document.getElementById("button-row"),
  ssid: document.getElementById("ssid"),
  password: document.getElementById("password"),
  togglePassword: document.getElementById("toggle-password"),
  ipAddress: document.getElementById("ip-address"),
  keepalive: document.getElementById("keepalive"),
  streamOverlay: document.getElementById("stream-overlay"),
  streamCamera: document.getElementById("stream-camera"),
  streamAudio: document.getElementById("stream-audio"),
  btnStart: document.getElementById("btn-start"),
  btnPreview: document.getElementById("btn-preview"),
  btnStop: document.getElementById("btn-stop"),
  logOutput: document.getElementById("log-output"),
  logStatus: document.getElementById("log-status"),
  ssidDropdown: document.getElementById("ssid-dropdown"),
  ssidDropdownBtn: document.getElementById("ssid-dropdown-btn"),
  tabBar: document.getElementById("tab-bar"),
  tabConnect: document.getElementById("tab-connect"),
  tabSettings: document.getElementById("tab-settings"),
  viewConnect: document.getElementById("view-connect"),
  viewSettings: document.getElementById("view-settings"),
  toolVersion: document.getElementById("tool-version"),
  toolAdbStatus: document.getElementById("tool-adb-status"),
  btnInstallAdb: document.getElementById("btn-install-adb"),
};

// ── Saved WiFi networks (ssid -> {psk, ip}) ──
let wifiNetworks = {};
const MAX_LOG_LINES = 500;
const LOG_FLUSH_INTERVAL_MS = 50;
let logEntries = [];
let pendingLogLines = [];
let logFlushTimer = null;
const CONNECTION_MODE_WIFI = "wifi";
const CONNECTION_MODE_USB = "usb";

function getConnectionMode() {
  return el.modeUsb.checked ? CONNECTION_MODE_USB : CONNECTION_MODE_WIFI;
}

function updateModeChipState() {
  const labels = el.connectionMode.querySelectorAll(".mode-chip");
  labels.forEach((label) => {
    const input = label.querySelector("input");
    label.classList.toggle("selected", !!input && input.checked);
  });
}

function applyConnectionMode(mode, options = {}) {
  const normalized = mode === CONNECTION_MODE_USB ? CONNECTION_MODE_USB : CONNECTION_MODE_WIFI;
  el.modeWifi.checked = normalized === CONNECTION_MODE_WIFI;
  el.modeUsb.checked = normalized === CONNECTION_MODE_USB;
  updateModeChipState();

  const isUsb = normalized === CONNECTION_MODE_USB;
  el.wifiSettings.classList.toggle("hidden", isUsb);
  el.usbModeHint.classList.toggle("hidden", !isUsb);

  el.buttonRow.classList.toggle("single-action", isUsb);
  el.btnPreview.classList.toggle("hidden", isUsb);
  el.btnStop.classList.toggle("hidden", isUsb);
  el.btnStart.textContent = isUsb ? "启动有线预览" : "启用";

  if (!options.skipSave) {
    debounceSave();
  }
}

// ── Build SSID dropdown (always shows all saved networks) ──
function buildSsidDropdown() {
  el.ssidDropdown.innerHTML = "";
  const ssids = Object.keys(wifiNetworks);
  if (ssids.length === 0) return;
  const currentSsid = el.ssid.value.trim();
  for (const ssid of ssids) {
    const item = document.createElement("div");
    item.className = "ssid-dropdown-item";
    if (ssid === currentSsid) {
      item.classList.add("active");
    }
    const net = wifiNetworks[ssid];
    item.textContent = ssid + (net.ip ? "  —  " + net.ip : "");
    item.addEventListener("mousedown", (e) => {
      e.preventDefault();
      el.ssid.value = ssid;
      el.password.value = net.psk;
      if (net.ip) setIpAddress(net.ip);
      hideDropdown();
      buildSsidDropdown();
    });
    el.ssidDropdown.appendChild(item);
  }
}

function showDropdown() {
  if (Object.keys(wifiNetworks).length > 0) {
    el.ssidDropdown.classList.add("visible");
  }
}

function hideDropdown() {
  el.ssidDropdown.classList.remove("visible");
}

// ── SSID dropdown button ──
el.ssidDropdownBtn.addEventListener("mousedown", (e) => {
  e.preventDefault();
  if (el.ssidDropdown.classList.contains("visible")) {
    hideDropdown();
  } else {
    showDropdown();
  }
});

// ── Hide dropdown on outside click ──
document.addEventListener("mousedown", (e) => {
  if (!el.ssidDropdownBtn.contains(e.target) && !el.ssidDropdown.contains(e.target)) {
    hideDropdown();
  }
});

// ── Debounced save config ──
let saveTimeout = null;
function debounceSave() {
  clearTimeout(saveTimeout);
  saveTimeout = setTimeout(() => {
    invoke("save_config", {
      connectionMode: getConnectionMode(),
      ssid: el.ssid.value.trim(),
      psk: el.password.value,
      ip: el.ipAddress.value.trim(),
      overlay: el.streamOverlay.checked,
      camera: el.streamCamera.checked,
      audio: el.streamAudio.checked,
    }).catch(() => {});
  }, 500);
}

el.ssid.addEventListener("input", debounceSave);
el.password.addEventListener("input", debounceSave);
el.ipAddress.addEventListener("input", debounceSave);
el.streamOverlay.addEventListener("change", debounceSave);
el.streamCamera.addEventListener("change", debounceSave);
el.streamAudio.addEventListener("change", debounceSave);
el.modeWifi.addEventListener("change", () => applyConnectionMode(CONNECTION_MODE_WIFI));
el.modeUsb.addEventListener("change", () => applyConnectionMode(CONNECTION_MODE_USB));

// ── Toggle password visibility ──
el.togglePassword.addEventListener("click", () => {
  const isPassword = el.password.type === "password";
  el.password.type = isPassword ? "text" : "password";
  el.togglePassword.textContent = isPassword ? "隐藏" : "显示";
});

// ── Log helper ──
function appendLog(text) {
  pendingLogLines.push(text);
  scheduleLogFlush();
}

function scheduleLogFlush() {
  if (logFlushTimer !== null) {
    return;
  }
  logFlushTimer = window.setTimeout(flushLogs, LOG_FLUSH_INTERVAL_MS);
}

function flushLogs() {
  logFlushTimer = null;
  if (pendingLogLines.length === 0) {
    return;
  }

  for (const line of pendingLogLines) {
    const lastEntry = logEntries[logEntries.length - 1];
    if (lastEntry && lastEntry.text === line) {
      lastEntry.count += 1;
    } else {
      logEntries.push({ text: line, count: 1 });
    }
  }
  pendingLogLines = [];

  if (logEntries.length > MAX_LOG_LINES) {
    logEntries = logEntries.slice(logEntries.length - MAX_LOG_LINES);
  }

  el.logOutput.value = logEntries
    .map((entry) => entry.count > 1 ? `${entry.text} [x${entry.count}]` : entry.text)
    .join("\n");
  if (logEntries.length > 0) {
    el.logOutput.value += "\n";
  }
  el.logOutput.scrollTop = el.logOutput.scrollHeight;
  el.logStatus.textContent = "（实时）";
}

// ── Set buttons enabled/disabled ──
function setButtonsEnabled(enabled) {
  el.btnStart.disabled = !enabled;
  el.btnPreview.disabled = !enabled;
  el.btnStop.disabled = !enabled;
}

// ── Update IP address input ──
function setIpAddress(ip) {
  el.ipAddress.value = ip || "";
}

// ── Event: log-message ──
listen("log-message", (event) => {
  appendLog(event.payload.message);
});

// ── Event: wifi-connected (IP from background thread) ──
listen("wifi-connected", (event) => {
  const ip = event.payload.ip;
  if (ip) {
    setIpAddress(ip);
    const ssid = el.ssid.value.trim();
    if (ssid) {
      invoke("save_wifi_network", { ssid: ssid, psk: el.password.value, ip: ip })
        .then(() => {
          wifiNetworks[ssid] = { psk: el.password.value, ip: ip };
          buildSsidDropdown();
        })
        .catch(() => {});
    }
  }
});

// ── Button: 启用 ──
el.btnStart.addEventListener("click", async () => {
  const mode = getConnectionMode();
  if (mode === CONNECTION_MODE_USB) {
    setButtonsEnabled(false);
    try {
      await invoke("preview", {
        connectionMode: CONNECTION_MODE_USB,
        ip: "",
        overlay: el.streamOverlay.checked,
        camera: el.streamCamera.checked,
        audio: el.streamAudio.checked,
      });
    } catch (e) {
      appendLog("错误: " + e);
    } finally {
      setButtonsEnabled(true);
    }
    return;
  }

  const ssid = el.ssid.value.trim();
  if (!ssid) {
    appendLog("请输入有效SSID");
    return;
  }

  setButtonsEnabled(false);
  try {
    await invoke("start_wifi", {
      ssid: ssid,
      password: el.password.value,
      keepalive: el.keepalive.checked,
    });
  } catch (e) {
    appendLog("错误: " + e);
  } finally {
    setButtonsEnabled(true);
  }
});

// ── Button: 停用 ──
el.btnStop.addEventListener("click", async () => {
  setButtonsEnabled(false);
  try {
    await invoke("stop_wifi");
  } catch (e) {
    appendLog("错误: " + e);
  } finally {
    setButtonsEnabled(true);
  }
});

// ── IP validation ──
function isValidIp(ip) {
  return /^((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)$/.test(ip);
}

// ── Button: 预览 ──
el.btnPreview.addEventListener("click", async () => {
  // Prefer saved IP for current SSID, fall back to input value
  const ssid = el.ssid.value.trim();
  const saved = ssid ? wifiNetworks[ssid] : null;
  const ip = (saved && saved.ip && isValidIp(saved.ip)) ? saved.ip : el.ipAddress.value.trim();

  if (!isValidIp(ip)) {
    appendLog("无效的IP地址，请先启动WIFI");
    return;
  }

  setButtonsEnabled(false);
  try {
    await invoke("preview", {
      connectionMode: getConnectionMode(),
      ip: ip,
      overlay: el.streamOverlay.checked,
      camera: el.streamCamera.checked,
      audio: el.streamAudio.checked,
    });
  } catch (e) {
    appendLog("错误: " + e);
  } finally {
    setButtonsEnabled(true);
  }
});

// ── 渲染效果参数（预览中实时生效）──
const EFFECT_FIELDS = [
  ["fx-overlay-alpha", "overlayAlpha", 2],
  ["fx-overlay-brightness", "overlayBrightness", 2],
  ["fx-overlay-scale", "overlayScale", 2],
  ["fx-base-brightness", "baseBrightness", 2],
  ["fx-dim-strength", "dimStrength", 2],
  ["fx-saturation-boost", "saturationBoost", 2],
  ["fx-black-key-low", "overlayBlackKeyLow", 2],
  ["fx-black-key-high", "overlayBlackKeyHigh", 2],
  ["fx-feather-power", "overlayFeatherPower", 2],
  ["fx-feather-radius", "overlayFeatherRadius", 0],
];
let effectSaveTimeout = null;

function effectPayload() {
  const payload = {};
  for (const [inputId, apiKey] of EFFECT_FIELDS) {
    payload[apiKey] = parseFloat(document.getElementById(inputId).value);
  }
  payload.baseRotateDeg = parseInt(document.getElementById("fx-base-rotate").value, 10);
  payload.overlayRotateDeg = parseInt(document.getElementById("fx-overlay-rotate").value, 10);
  return payload;
}

function updateEffectLabels() {
  for (const [inputId, , decimals] of EFFECT_FIELDS) {
    const value = document.getElementById(inputId).value;
    document.getElementById(inputId + "-value").textContent =
      decimals === 0 ? String(parseInt(value, 10)) : parseFloat(value).toFixed(decimals);
  }
}

function scheduleEffectSave() {
  updateEffectLabels();
  clearTimeout(effectSaveTimeout);
  effectSaveTimeout = setTimeout(() => {
    invoke("set_effect", effectPayload()).catch((e) => appendLog("保存效果参数失败: " + e));
  }, 300);
}

for (const [inputId] of EFFECT_FIELDS) {
  document.getElementById(inputId).addEventListener("input", scheduleEffectSave);
}
document.getElementById("fx-base-rotate").addEventListener("change", scheduleEffectSave);
document.getElementById("fx-overlay-rotate").addEventListener("change", scheduleEffectSave);

// ── 恢复默认参数 ──
document.getElementById("btn-reset-effect").addEventListener("click", async () => {
  try {
    const effects = await invoke("reset_effect");
    for (const [inputId, apiKey] of EFFECT_FIELDS) {
      if (typeof effects[apiKey] === "number") {
        document.getElementById(inputId).value = effects[apiKey];
      }
    }
    if (typeof effects.baseRotateDeg === "number") {
      document.getElementById("fx-base-rotate").value = String(effects.baseRotateDeg);
    }
    if (typeof effects.overlayRotateDeg === "number") {
      document.getElementById("fx-overlay-rotate").value = String(effects.overlayRotateDeg);
    }
    updateEffectLabels();
    appendLog("渲染效果已恢复默认值");
  } catch (e) {
    appendLog("恢复默认失败: " + e);
  }
});

// ── 页面切换（连接 / 设置）──
function showView(view) {
  const isSettings = view === "settings";
  el.viewConnect.classList.toggle("hidden", isSettings);
  el.viewSettings.classList.toggle("hidden", !isSettings);
  el.tabConnect.classList.toggle("selected", !isSettings);
  el.tabSettings.classList.toggle("selected", isSettings);
  document.body.classList.toggle("view-settings", isSettings);
  document.body.classList.toggle("view-connect", !isSettings);
  try {
    localStorage.setItem("gscp.view", isSettings ? "settings" : "connect");
  } catch (e) { /* WebView 存储不可用时忽略 */ }
  requestResize();
}

// ── 窗口外框高度随内容自适应 ──
// 测量时临时去掉容器的 min-height，得到内容的自然高度；
// 原生命令会补偿标题栏高度并保持宽度不变。
function requestResize() {
  setTimeout(() => {
    const container = document.querySelector(".container");
    if (!container) return;
    const prevMin = container.style.minHeight;
    container.style.minHeight = "0px";
    const target = Math.ceil(container.getBoundingClientRect().height);
    container.style.minHeight = prevMin;
    invoke("resize_height_to", { contentHeight: target, innerHeight: window.innerHeight }).catch(() => {});
  }, 60);
}

el.tabConnect.addEventListener("click", () => showView("connect"));
el.tabSettings.addEventListener("click", () => showView("settings"));

// ── 运行环境 ──
async function refreshToolStatus() {
  try {
    const status = await invoke("tool_status");
    if (status.adbAvailable) {
      el.toolAdbStatus.textContent = "adb: " + status.adbPath;
      el.btnInstallAdb.classList.add("hidden");
    } else {
      el.toolAdbStatus.textContent = "未找到 adb（预览前将自动下载 platform-tools）";
      el.btnInstallAdb.classList.remove("hidden");
    }
  } catch (e) {
    el.toolAdbStatus.textContent = "adb 状态检查失败";
  }
}

el.btnInstallAdb.addEventListener("click", () => {
  el.toolAdbStatus.textContent = "正在下载 platform-tools ...";
  invoke("install_platform_tools")
    .then(() => {
      // 下载在后台线程执行，轮询状态
      let tries = 0;
      const timer = setInterval(() => {
        tries += 1;
        invoke("tool_status").then((status) => {
          if (status.adbAvailable) {
            clearInterval(timer);
            refreshToolStatus();
          } else if (tries > 120) {
            clearInterval(timer);
          }
        });
      }, 1000);
    })
    .catch((e) => appendLog("下载失败: " + e));
});

// ── Init: load config and wifi list ──
async function init() {
  try {
    const version = await invoke("get_app_version");
    el.toolVersion.textContent = "v" + version;
  } catch (e) { /* ignore */ }

  try {
    const config = await invoke("get_config");
    applyConnectionMode(config.connectionMode || CONNECTION_MODE_WIFI, { skipSave: true });
    el.ssid.value = config.ssid || "";
    el.password.value = config.psk || "";
    if (config.ip) {
      setIpAddress(config.ip);
    }
    el.streamOverlay.checked = config.overlay !== false;
    el.streamCamera.checked = config.camera !== false;
    el.streamAudio.checked = config.audio !== false;
  } catch (e) {
    applyConnectionMode(CONNECTION_MODE_WIFI, { skipSave: true });
  }

  try {
    const list = await invoke("get_wifi_list");
    for (const item of list) {
      wifiNetworks[item.ssid] = { psk: item.psk, ip: item.ip || "" };
    }
    buildSsidDropdown();
  } catch (e) {
    // Ignore errors
  }

  try {
    const effects = await invoke("get_effect");
    for (const [inputId, apiKey] of EFFECT_FIELDS) {
      if (typeof effects[apiKey] === "number") {
        document.getElementById(inputId).value = effects[apiKey];
      }
    }
    if (typeof effects.baseRotateDeg === "number") {
      document.getElementById("fx-base-rotate").value = String(effects.baseRotateDeg);
    }
    if (typeof effects.overlayRotateDeg === "number") {
      document.getElementById("fx-overlay-rotate").value = String(effects.overlayRotateDeg);
    }
    updateEffectLabels();
  } catch (e) {
    // 演示模式下忽略
  }

  refreshToolStatus();
  if (platform === "none") {
    appendLog("（演示模式：未检测到原生后端）");
  }

  let savedView = null;
  try {
    savedView = localStorage.getItem("gscp.view");
  } catch (e) { /* ignore */ }
  showView(savedView === "settings" ? "settings" : "connect");
}

window.addEventListener("load", requestResize);
init();
