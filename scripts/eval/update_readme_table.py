"""Replace the benchmark table in README.md (between the RESULTS_TABLE markers) with scripts/eval/collect_results.py output."""
import re
import subprocess
import sys

table = subprocess.check_output([sys.executable, "scripts/eval/collect_results.py", "--results_dir", sys.argv[1] if len(sys.argv) > 1 else "results"], text=True).strip()
readme = open("README.md").read()
new = re.sub(r"(<!-- RESULTS_TABLE_START -->\n).*?(<!-- RESULTS_TABLE_END -->)", lambda m: m.group(1) + table + "\n" + m.group(2), readme, flags=re.S)
open("README.md", "w").write(new)
print(table)
