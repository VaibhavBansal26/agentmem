# Releasing memorank to PyPI

Publishing uses PyPI Trusted Publishing: GitHub Actions proves its identity to PyPI, so no API token is created or stored.

## One-time setup (done by the project owner)

1. On **https://test.pypi.org** and **https://pypi.org**, open Account settings, then Publishing, and add a *pending publisher*:
   - PyPI project name: `memorank`
   - Owner: `VaibhavBansal26`
   - Repository: `memorank`
   - Workflow filename: `publish.yml`
   - Environment name: `testpypi` on TestPyPI, `pypi` on PyPI
2. In the GitHub repo settings, create the environments `testpypi` and `pypi`. For `pypi`, add yourself as a required reviewer so
   a human approves every real release.
3. Check that the name is still free the day you publish.

## Each release

1. Update `version` in `pyproject.toml`, commit, and make sure CI is green.
2. Rehearse: Actions, `publish`, Run workflow, target `testpypi`. Then in a clean environment:
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ memorank`
3. Publish: create a GitHub release whose tag is `v<version>` (for example `v0.1.0`). The workflow refuses to run if the tag and
   the package version disagree, and the `pypi` environment waits for your approval.
