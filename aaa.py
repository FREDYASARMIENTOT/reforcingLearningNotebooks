import subprocess, os, sys

root = os.path.abspath("d:\\ReforcingLearning")
out_path = os.path.join(root, "z_result.txt")

def r(cmd):
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip() or "(empty)"

lines = [
    "=== GIT STATUS ===",
    r(["git", "status", "--short"]),
    "",
    "=== GIT BRANCH ===",
    r(["git", "branch"]),
    "",
    "=== GIT REMOTE ===",
    r(["git", "remote", "-v"]),
    "",
    "=== GIT LOG ===",
    r(["git", "log", "--oneline", "-5"]),
    "",
    "=== GIT ROOT ===",
    r(["git", "rev-parse", "--show-toplevel"]),
]

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print("DONE")
sys.stdout.flush()