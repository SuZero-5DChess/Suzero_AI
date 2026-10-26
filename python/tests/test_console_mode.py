from __future__ import annotations

from suzero_5dchess import console_mode


class FakeFunction:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class FakeKernel32:
    def __init__(self, mode: int, *, set_succeeds: bool = True) -> None:
        self.mode = mode
        self.set_succeeds = set_succeeds
        self.set_modes: list[int] = []
        self.GetStdHandle = FakeFunction(lambda _kind: 123)
        self.GetConsoleMode = FakeFunction(self._get_mode)
        self.SetConsoleMode = FakeFunction(self._set_mode)

    def _get_mode(self, handle, output) -> int:
        assert handle == 123
        output._obj.value = self.mode
        return 1

    def _set_mode(self, handle, mode) -> int:
        assert handle == 123
        value = int(mode.value) if hasattr(mode, "value") else int(mode)
        self.set_modes.append(value)
        if not self.set_succeeds:
            return 0
        self.mode = value
        return 1


def test_disable_quick_edit_changes_and_restores_console_mode(monkeypatch) -> None:
    unrelated_flag = 0x0001
    original = unrelated_flag | console_mode.ENABLE_QUICK_EDIT_MODE
    kernel32 = FakeKernel32(original)
    monkeypatch.setattr(console_mode, "_load_kernel32", lambda: kernel32)

    restore = console_mode.disable_quick_edit()

    assert restore is not None
    assert kernel32.mode == unrelated_flag | console_mode.ENABLE_EXTENDED_FLAGS
    restore()
    restore()
    assert kernel32.mode == original
    assert kernel32.set_modes == [
        unrelated_flag | console_mode.ENABLE_EXTENDED_FLAGS,
        original,
    ]


def test_disable_quick_edit_is_noop_without_console(monkeypatch) -> None:
    monkeypatch.setattr(console_mode, "_load_kernel32", lambda: None)

    assert console_mode.disable_quick_edit() is None


def test_disable_quick_edit_preserves_an_already_safe_mode(monkeypatch) -> None:
    original = console_mode.ENABLE_EXTENDED_FLAGS | 0x0001
    kernel32 = FakeKernel32(original)
    monkeypatch.setattr(console_mode, "_load_kernel32", lambda: kernel32)

    assert console_mode.disable_quick_edit() is None
    assert kernel32.mode == original
    assert kernel32.set_modes == []


def test_disable_quick_edit_does_not_offer_restore_after_failure(monkeypatch) -> None:
    kernel32 = FakeKernel32(
        console_mode.ENABLE_QUICK_EDIT_MODE,
        set_succeeds=False,
    )
    monkeypatch.setattr(console_mode, "_load_kernel32", lambda: kernel32)

    assert console_mode.disable_quick_edit() is None
    assert kernel32.set_modes == [console_mode.ENABLE_EXTENDED_FLAGS]
