// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// Immediate Ctrl+C feedback for a call that blocks the Python main thread.
//
// `run_selfplay` releases the GIL and returns only after the whole batch has
// been played, so Python's SIGINT handler cannot run meanwhile. Without a
// native handler a Ctrl+C during self-play looks completely ignored until the
// batch ends, even though the progress bar keeps moving. This guard covers
// exactly that window:
//
//   * the first press prints a notice straight to stderr, so the user learns
//     the signal was received while the batch keeps running;
//   * a second press terminates the process, matching the trainer's documented
//     "press again to abort immediately";
//   * the signal is never swallowed: the native handler chains to the previous
//     handler, so CPython still records the SIGINT and the cooperative stop in
//     python/suzero_5dchess/interrupts.py fires when the call returns.
//
// This is not a general signal facility. Install one guard per blocking call,
// on one thread at a time, and destroy it before returning to Python.
// This header is private to the pybind11 adapter.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

namespace suzero::chess5d::internal {

inline constexpr char kInterruptNotice[] =
    "\n[Ctrl+C] 已收到中断信号：正在进行的对局会跑完，不再开始新对局；"
    "样本写入文件并保存后退出，下次从该进度继续；再按一次 Ctrl+C 立即中止。\n";

inline constexpr char kInterruptAbort[] =
    "\n[Ctrl+C] 再次收到中断信号，立即中止（本轮进度可能丢失）。\n";

/// Set on the first press. The self-play scheduler stops handing out new games
/// while games already in flight run to completion.
inline std::atomic<bool> g_stopRequested{false};

/// Press count of the active guard. Written from the OS handler thread(s).
inline std::atomic<int> g_interruptPresses{0};

/// Guards every handler entry: after the guard is destroyed a late signal must
/// fall through to CPython instead of being counted or printed here.
inline std::atomic<bool> g_interruptActive{false};

inline bool stopRequested() noexcept {
  return g_stopRequested.load(std::memory_order_acquire);
}

/// Writes UTF-8 text to stderr, readable both on a console and in a log file.
inline void writeInterruptMessage(const char* text) noexcept {
#ifdef _WIN32
  const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
  if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
    return;
  }
  DWORD mode = 0;
  if (GetConsoleMode(handle, &mode) != 0) {
    // Console handles consume UTF-16, not the process code page.
    wchar_t wide[256];
    const int length = MultiByteToWideChar(
        CP_UTF8, 0, text, -1, wide,
        static_cast<int>(sizeof(wide) / sizeof(wide[0])));
    if (length > 1) {
      DWORD written = 0;
      WriteConsoleW(handle, wide, static_cast<DWORD>(length - 1), &written,
                    nullptr);
    }
    return;
  }
  // Redirected stderr (log file, pipe): keep the UTF-8 bytes unchanged.
  DWORD written = 0;
  WriteFile(handle, text, static_cast<DWORD>(std::strlen(text)), &written,
            nullptr);
#else
  const std::size_t length = std::strlen(text);
  std::size_t offset = 0;
  while (offset < length) {
    const ssize_t written =
        ::write(STDERR_FILENO, text + offset, length - offset);
    if (written <= 0) {
      return;
    }
    offset += static_cast<std::size_t>(written);
  }
#endif
}

#ifdef _WIN32

inline BOOL WINAPI consoleCtrlHandler(DWORD type) {
  if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT) {
    return FALSE;
  }
  if (!g_interruptActive.load(std::memory_order_acquire)) {
    return FALSE;
  }
  const bool ctrl_c = type == CTRL_C_EVENT;
  const int presses =
      g_interruptPresses.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (presses == 1) {
    g_stopRequested.store(true, std::memory_order_release);
    writeInterruptMessage(kInterruptNotice);
    if (ctrl_c) {
      // FALSE chains to the handlers registered earlier (CPython's), which
      // records the SIGINT for the main thread. Returning TRUE here would
      // swallow it and Python's cooperative stop would never fire.
      return FALSE;
    }
    // A parent sends CTRL_BREAK to a detached child process group. Returning
    // TRUE keeps the default SIGBREAK termination from killing the child
    // mid-game; the native scheduler observes g_stopRequested instead.
    return TRUE;
  }
  writeInterruptMessage(kInterruptAbort);
  TerminateProcess(GetCurrentProcess(), 130);
  return TRUE;
}

#else

inline struct sigaction g_previousInterruptAction {};

inline void posixInterruptHandler(int signum) {
  const int presses =
      g_interruptPresses.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (presses == 1) {
    g_stopRequested.store(true, std::memory_order_release);
    writeInterruptMessage(kInterruptNotice);
    // Chain to the previous handler (CPython's), so the interpreter still
    // sees SIGINT and applies the cooperative stop after the call returns.
    if ((g_previousInterruptAction.sa_flags & SA_SIGINFO) != 0) {
      if (g_previousInterruptAction.sa_sigaction != nullptr) {
        siginfo_t info{};
        info.si_signo = signum;
        g_previousInterruptAction.sa_sigaction(signum, &info, nullptr);
      }
    } else if (g_previousInterruptAction.sa_handler != SIG_DFL &&
               g_previousInterruptAction.sa_handler != SIG_IGN) {
      g_previousInterruptAction.sa_handler(signum);
    }
    return;
  }
  writeInterruptMessage(kInterruptAbort);
  _exit(130);
}

#endif

/// Installs the Ctrl+C handler for the lifetime of one blocking call.
class ConsoleInterruptGuard {
 public:
  ConsoleInterruptGuard() noexcept {
    g_interruptPresses.store(0, std::memory_order_relaxed);
    g_stopRequested.store(false, std::memory_order_release);
    g_interruptActive.store(true, std::memory_order_release);
#ifdef _WIN32
    installed_ = SetConsoleCtrlHandler(consoleCtrlHandler, TRUE) != 0;
#else
    struct sigaction action {};
    action.sa_handler = posixInterruptHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    installed_ = sigaction(SIGINT, &action, &g_previousInterruptAction) == 0;
#endif
  }

  ~ConsoleInterruptGuard() {
    // Deactivate first: a signal racing with the restore below must go to
    // CPython untouched.
    g_interruptActive.store(false, std::memory_order_release);
    g_stopRequested.store(false, std::memory_order_release);
#ifdef _WIN32
    if (installed_) {
      SetConsoleCtrlHandler(consoleCtrlHandler, FALSE);
    }
#else
    if (installed_) {
      sigaction(SIGINT, &g_previousInterruptAction, nullptr);
    }
#endif
  }

  ConsoleInterruptGuard(const ConsoleInterruptGuard&) = delete;
  ConsoleInterruptGuard& operator=(const ConsoleInterruptGuard&) = delete;

 private:
  bool installed_ = false;
};

}  // namespace suzero::chess5d::internal
