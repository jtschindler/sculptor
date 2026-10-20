"""Tests for `sculptor.logging_config`."""

import logging

import pytest

import sculptor
from sculptor import logging_config as lc
from sculptor import model as scmod
from sculptor import parameter as scpar
from sculptor import prior as scpri


@pytest.fixture(autouse=True)
def default_logging(monkeypatch):
    """Every test starts and ends with the import-time default setup."""

    monkeypatch.delenv(lc.LOG_LEVEL_ENV_VAR, raising=False)
    monkeypatch.delenv(lc.LOG_TO_FILE_ENV_VAR, raising=False)
    sculptor.configure_logging()
    yield
    monkeypatch.delenv(lc.LOG_LEVEL_ENV_VAR, raising=False)
    lc.reset_logging()
    sculptor.configure_logging()


def _own_handlers(name):
    return [
        h
        for h in logging.getLogger(name).handlers
        if getattr(h, lc._OWN_HANDLER_ATTR, False)
    ]


def test_import_configures_only_the_package_loggers():
    for name in lc.PACKAGE_LOGGERS:
        logger = logging.getLogger(name)
        assert logger.level == logging.INFO
        assert logger.propagate is False
        assert len(_own_handlers(name)) == 2
    assert not any(
        getattr(h, lc._OWN_HANDLER_ATTR, False) for h in logging.getLogger().handlers
    )


def test_info_to_stdout_warning_to_stderr(capsys):
    log = logging.getLogger("sculptor.some_module")
    log.debug("not shown")
    log.info("progress")
    log.warning("careful")

    out, err = capsys.readouterr()
    assert out == "sculptor.some_module - INFO: progress\n"
    assert err == "sculptor.some_module - WARNING: careful\n"


def test_extensions_logger_is_configured_too(capsys):
    logging.getLogger("sculptor_extensions.qso_autofit").info("hello")
    assert "sculptor_extensions.qso_autofit - INFO: hello" in capsys.readouterr().out


def test_repeated_calls_do_not_duplicate_handlers(capsys):
    sculptor.configure_logging()
    sculptor.configure_logging()

    assert len(_own_handlers("sculptor")) == 2
    logging.getLogger("sculptor").info("once")
    assert capsys.readouterr().out.count("once") == 1


@pytest.mark.parametrize("level", [logging.WARNING, "warning", " Warning "])
def test_level_argument(capsys, level):
    sculptor.configure_logging(level=level)
    logging.getLogger("sculptor").info("hidden")

    assert logging.getLogger("sculptor").level == logging.WARNING
    assert capsys.readouterr().out == ""


def test_level_from_env_var(monkeypatch):
    monkeypatch.setenv(lc.LOG_LEVEL_ENV_VAR, "debug")
    sculptor.configure_logging()
    assert logging.getLogger("sculptor").level == logging.DEBUG

    monkeypatch.setenv(lc.LOG_LEVEL_ENV_VAR, "loud")
    with pytest.raises(ValueError, match=lc.LOG_LEVEL_ENV_VAR):
        sculptor.configure_logging()


def test_log_file_and_backup(tmp_path):
    sculptor.configure_logging(log_to_file=True, local_path=tmp_path)
    logging.getLogger("sculptor").info("first run")
    sculptor.configure_logging(log_to_file=True, local_path=tmp_path)
    logging.getLogger("sculptor").info("second run")
    for handler in _own_handlers("sculptor"):
        handler.flush()

    log_dir = tmp_path / "logs"
    assert "second run" in (log_dir / "sculptor.log").read_text()
    backups = list(log_dir.glob("sculptor_*.log.bak"))
    assert len(backups) == 1 and "first run" in backups[0].read_text()


def test_reset_hands_messages_to_the_application(caplog):
    lc.reset_logging()

    assert _own_handlers("sculptor") == []
    assert logging.getLogger("sculptor").propagate is True
    with caplog.at_level(logging.INFO):
        logging.getLogger("sculptor.model").info("to root")
    assert "to root" in caplog.text


def test_warn_once(capsys):
    log = logging.getLogger("sculptor.warn_once_test")
    for _ in range(3):
        lc.warn_once(log, "only once")

    assert capsys.readouterr().err.count("only once") == 1


def test_all_logging_disabled(capsys):
    with lc.all_logging_disabled():
        logging.getLogger("sculptor").warning("swallowed")
    logging.getLogger("sculptor").warning("shown")

    err = capsys.readouterr().err
    assert "swallowed" not in err and "shown" in err


def test_package_messages_go_through_logging(capsys):
    """A former `print` in FitModel now arrives as a log record."""

    model = scmod.FitModel(
        parameters={
            "a": scpar.FitParameter("a", 1.0, prior=scpri.UniformPrior("a", 0, 2)),
            "b": scpar.FitParameter("b", 5.0),
        }
    )
    model.nwalkers = 4
    model.initialize_positions()

    out, err = capsys.readouterr()
    assert "sculptor.model - INFO: Sampled prior for a" in out
    assert "sculptor.model - WARNING: No prior specified for b" in err
