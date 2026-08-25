import logging
import subprocess
from types import SimpleNamespace

import pytest

from mre import DocFormat
from mre.hwp_convert import (
    HwpConversionError,
    LibreOfficeNotAvailableError,
    convert_hwp,
)


def _fake_source(tmp_path):
    src = tmp_path / "report.hwp"
    src.write_bytes(b"not a real hwp -- soffice is mocked out in these tests")
    return src


def test_convert_hwp_rejects_unsupported_target(tmp_path):
    src = _fake_source(tmp_path)
    with pytest.raises(ValueError):
        convert_hwp(src, target=DocFormat.HTML)


def test_convert_hwp_missing_source_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_hwp(tmp_path / "does_not_exist.hwp")


def test_convert_hwp_logs_warning_about_best_effort_fidelity(tmp_path, monkeypatch, caplog):
    src = _fake_source(tmp_path)
    out_path = tmp_path / "report.docx"

    def _fake_run(cmd, **kwargs):
        out_path.write_bytes(b"fake docx")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    with caplog.at_level(logging.WARNING, logger="mre.hwp_convert"):
        convert_hwp(src)

    assert any("best-effort" in r.message or "best-effort" in r.getMessage() for r in caplog.records)


def test_convert_hwp_soffice_not_found_raises(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)

    def _fake_run(cmd, **kwargs):
        raise FileNotFoundError("soffice")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    with pytest.raises(LibreOfficeNotAvailableError):
        convert_hwp(src)


def test_convert_hwp_timeout_raises_conversion_error(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)

    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    with pytest.raises(HwpConversionError):
        convert_hwp(src, timeout=5)


def test_convert_hwp_nonzero_returncode_raises_conversion_error(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)

    def _fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="some soffice failure")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    with pytest.raises(HwpConversionError, match="some soffice failure"):
        convert_hwp(src)


def test_convert_hwp_missing_output_file_raises_even_with_zero_returncode(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)

    def _fake_run(cmd, **kwargs):
        # returncode 0 but soffice silently didn't produce the expected file
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    with pytest.raises(HwpConversionError):
        convert_hwp(src)


def test_convert_hwp_success_returns_output_path(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)
    out_path = tmp_path / "report.docx"
    captured_cmd = {}

    def _fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        out_path.write_bytes(b"fake docx")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    result = convert_hwp(src, target=DocFormat.DOCX)

    assert result == out_path
    cmd = captured_cmd["cmd"]
    assert cmd[0] == "soffice"
    assert "--convert-to" in cmd and "docx" in cmd
    assert str(src) in cmd


def test_convert_hwp_pdf_target(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)
    out_path = tmp_path / "report.pdf"

    def _fake_run(cmd, **kwargs):
        out_path.write_bytes(b"fake pdf")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    result = convert_hwp(src, target=DocFormat.PDF)

    assert result == out_path


def test_convert_hwp_creates_outdir_if_missing(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)
    outdir = tmp_path / "nested" / "output"
    out_path = outdir / "report.docx"

    def _fake_run(cmd, **kwargs):
        out_path.write_bytes(b"fake docx")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    result = convert_hwp(src, target=DocFormat.DOCX, outdir=outdir)

    assert result == out_path
    assert outdir.is_dir()


def test_convert_hwp_uses_custom_soffice_bin(tmp_path, monkeypatch):
    src = _fake_source(tmp_path)
    out_path = tmp_path / "report.docx"
    captured_cmd = {}

    def _fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        out_path.write_bytes(b"fake docx")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mre.hwp_convert.subprocess.run", _fake_run)

    convert_hwp(src, soffice_bin="/opt/libreoffice/program/soffice")

    assert captured_cmd["cmd"][0] == "/opt/libreoffice/program/soffice"
