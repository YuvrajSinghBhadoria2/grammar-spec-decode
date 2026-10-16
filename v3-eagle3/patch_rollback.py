import sys

p = "/root/grammar-spec/eagle3_grammar_b1.py"
src = open(p).read()
old = """                if m is not None and nadv > 0:
                    try:
                        m.rollback(nadv)
                    except Exception:
                        pass"""
new = """                if m is not None and nadv > 0:
                    try:
                        m.rollback(nadv)
                    except Exception as _e:
                        with open("/root/grammar-spec/rollback_fail.log", "a") as _f:
                            _f.write("nadv=%d err=%s: %s\\n" % (nadv, type(_e).__name__, _e))"""
if old not in src:
    # Already patched?
    if "rollback_fail.log" in src:
        print("ALREADY_PATCHED")
        sys.exit(0)
    print("PATTERN_NOT_FOUND")
    sys.exit(1)
src = src.replace(old, new)
open(p, "w").write(src)
print("PATCHED")
