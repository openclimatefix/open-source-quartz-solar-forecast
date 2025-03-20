"""Utils used in tests."""


from click.testing import CliRunner


def run_click_command(main_func, cmd_args: list[str]):
    """Run a click command in a test-fiendly way."""
    runner = CliRunner()

    result = runner.invoke(main_func, cmd_args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by click's test runner.
    print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception:
        raise result.exception

    assert result.exit_code == 0

    return result
