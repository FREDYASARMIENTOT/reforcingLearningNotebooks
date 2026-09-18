import subprocess, os

root = os.path.abspath(r"D:\ReforcingLearning")
out_path = os.path.join(root, "git_audit_result.txt")

def run_git(args):
    result = subprocess.run(["git"] + args, cwd=root, capture_output=True, text=True)
    out = result.stdout.strip()
    err = result.stderr.strip()
    return out if out else ("(empty)" if not err else "STDERR: " + err)

lines = []
lines.append("=== GIT STATUS ===")
lines.append(run_git(["status", "--short"]))
lines.append("")
lines.append("=== GIT BRANCH ===")
lines.append(run_git(["branch"]))
lines.append("")
lines.append("=== GIT REMOTE ===")
lines.append(run_git(["remote", "-v"]))
lines.append("")
lines.append("=== GIT LOG ===")
lines.append(run_git(["log", "--oneline", "-5"]))
lines.append("")
lines.append("=== GIT ROOT ===")
lines.append(run_git(["rev-parse", "--show-toplevel"]))

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print("DONE")
import sys
sys.stdout.flush()