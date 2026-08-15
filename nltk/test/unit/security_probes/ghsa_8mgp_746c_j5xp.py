"""GHSA-8mgp-746c-j5xp [high] -- Model-artifact APIs bypass pathsec and touch files outside allowed roots"""

import os
import pathlib
import shutil
import tempfile

from ._base import FIXED, STATIC, VULNERABLE, probe


@probe("GHSA-8mgp-746c-j5xp")
def _model_artifact_apis():
    """The sandbox chokepoint waved through any URL-shaped string.

    ``validate_path`` returned success for anything containing ``"://"`` with an
    http/https/ftp scheme, so ``http://../../etc/passwd`` -- a kernel-level
    traversal, not a host -- escaped every allowed root, and the low-level model
    ``save``/``load`` APIs opened caller paths directly on top of that.

    Two checks. The URL primitive is also the teeth: with enforcement off the
    guard degrades to a warning and returns, so the probe flips to VULNERABLE.
    The end-to-end save is gated by a negative control proving the write target
    is genuinely outside the sandbox -- a temp dir is not (the private system
    temp *is* an allowed root on macOS), so the target is a fresh ``$HOME`` dir.
    """
    import nltk
    import nltk.pathsec as pathsec

    # 1) The advisory itself: a URL must not authorize a filesystem path. Under
    #    ENFORCE this raises; with the guard removed it returns -> VULNERABLE.
    url = "http://../../../../etc/passwd"
    try:
        pathsec.validate_path(url)
        return (
            VULNERABLE,
            "validate_path authorized a URL-prefixed traversal (%r)" % url,
        )
    except PermissionError:
        pass

    # 2) End-to-end: the model save/load APIs must refuse an outside path too.
    saved_paths = nltk.data.path[:]
    sandbox = tempfile.mkdtemp()
    outside_dir = pathlib.Path.home() / (".nltk_ghsa8mgp_probe_%d" % os.getpid())
    outside = str(outside_dir / "model.json")
    try:
        nltk.data.path[:] = [sandbox]
        pathsec._ALLOWED_ROOTS_CACHE = None
        pathsec._LAST_DATA_PATHS = None
        outside_dir.mkdir(exist_ok=True)

        # Negative control: the target must be genuinely refused, or the save
        # result below would be meaningless. If pathsec itself allows it (target
        # inside a root, or enforcement off), we cannot measure the sink here --
        # fall back to the URL-primitive verdict above.
        try:
            with pathsec.open(outside, "w"):
                pass
            if os.path.exists(outside):
                os.remove(outside)
            return FIXED, "URL path check fails closed (save sink not measurable here)"
        except PermissionError:
            pass  # good: genuinely outside the sandbox

        from nltk.tag.perceptron import AveragedPerceptron

        ap = AveragedPerceptron()
        ap.weights = {"f": {"t": 1.0}}
        try:
            ap.save(outside)
        except PermissionError:
            return FIXED, "URL path check and model save both fail closed"
        leaked = os.path.exists(outside)
        if os.path.exists(outside):
            os.remove(outside)
        if leaked:
            return VULNERABLE, "AveragedPerceptron.save wrote outside allowed root"
        return FIXED, "URL path check and model save both fail closed"
    finally:
        nltk.data.path[:] = saved_paths
        pathsec._ALLOWED_ROOTS_CACHE = None
        pathsec._LAST_DATA_PATHS = None
        shutil.rmtree(outside_dir, ignore_errors=True)
        shutil.rmtree(sandbox, ignore_errors=True)
