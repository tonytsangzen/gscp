// 平台桥接层：桌面端（Tauri IPC）与 Android 端（JavascriptInterface）
// 提供一致的 invoke/listen 接口。
(function () {
  "use strict";

  if (typeof window.__TAURI__ !== "undefined" && window.__TAURI__.core) {
    // 桌面端：Tauri 2 withGlobalTauri
    window.gscpBridge = {
      platform: "tauri",
      invoke: window.__TAURI__.core.invoke,
      listen: function (event, handler) {
        return window.__TAURI__.event.listen(event, handler);
      },
    };
    return;
  }

  if (typeof window.gscpAndroid !== "undefined") {
    // Android 端：MainActivity 注入的 JavascriptInterface
    // gscpAndroid.invoke(name, jsonArgs) -> Promise 由下方适配
    var pending = {};
    var seq = 0;
    window.__gscpAndroidReply = function (id, ok, value) {
      var entry = pending[id];
      if (!entry) return;
      delete pending[id];
      if (ok) entry.resolve(value);
      else entry.reject(value);
    };
    window.gscpBridge = {
      platform: "android",
      invoke: function (name, args) {
        return new Promise(function (resolve, reject) {
          var id = String(++seq);
          pending[id] = { resolve: resolve, reject: reject };
          window.gscpAndroid.invoke(name, JSON.stringify(args || {}), id);
        });
      },
      listen: function (event, handler) {
        // Android 侧通过 gscpAndroid.emit(event, jsonPayload) 推送
        var listeners = (window.__gscpListeners = window.__gscpListeners || {});
        (listeners[event] = listeners[event] || []).push(handler);
        window.gscpAndroid.listen && window.gscpAndroid.listen(event);
        return Promise.resolve(function () {
          listeners[event] = (listeners[event] || []).filter(function (h) { return h !== handler; });
        });
      },
    };
    window.__gscpAndroidEmit = function (event, payloadJson) {
      var listeners = (window.__gscpListeners || {})[event] || [];
      var payload;
      try { payload = JSON.parse(payloadJson); } catch (e) { payload = payloadJson; }
      listeners.forEach(function (h) { h({ payload: payload }); });
    };
    return;
  }

  // 浏览器直接打开（开发预览）：提供空实现
  console.warn("gscpBridge 未找到原生后端，进入无后端演示模式");
  window.gscpBridge = {
    platform: "none",
    invoke: function (name) {
      if (name === "get_effect") {
        return Promise.resolve({});
      }
      if (name === "tool_status") {
        return Promise.resolve({ adbAvailable: false, adbPath: null });
      }
      return Promise.reject("无后端");
    },
    listen: function () {
      return Promise.resolve(function () {});
    },
  };
})();
