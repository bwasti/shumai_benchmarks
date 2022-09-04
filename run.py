import os
import math
import time
import sys

cmds = [
  ["tensorflow (js):  ", "node test_tf.js",u"\u001b[31;1m"],
  ["pytorch (python): ", "python test_pt.py",u"\u001b[35;1m"],
  ["shumai (js):      ", "bun test_sm.js", u"\u001b[36;1m"],
]
if len(sys.argv) == 2:
  if sys.argv[1] == "pytorch":
    cmds = [cmds[1], cmds[2]]
  elif sys.argv[1] == "tensorflow":
    cmds = [cmds[0], cmds[2]]

def run(desc, op, sizes):
  print(f"\033[4m\033[1m{desc}\033[0m\n")
  for n, iters in sizes:
    warmup = iters // 10
    print(f" N={n} (run {iters} times)")
    vals = []
    for name, cmd, color in cmds:
      full_cmd = f"{cmd} {op} {n} {iters} {warmup} 2>/dev/null"
      print(color, end='')
      print(f"   {name} ", end='', flush=True)
      t0 = time.time()
      out = os.popen(full_cmd).read().strip()

      try:
        v = float(out.split()[0].replace('K',''))
        vals.append(v)
      except:
        pass
      t1 = time.time()
      print(f"{out if out else 'error'} ({t1-t0:.2f} seconds total, including init and warmup)", end='')
      print("\033[2m\t", full_cmd, u"\u001b[0m")
    if len(vals) == 2:
      print(f"   \033[1mdifference:        {vals[1]/vals[0]:.2f}x\033[0m")
    print()

run("N-wide vector pointwise addition", "pw", [(32, 20000), (1024, 1000), (32 * 1024, 1000)])
print()
run("NxN matrix multiplication", "mm", [(64, 10000), (128, 10000), (1024, 1000)])
print()
run("N-wide hidden layer, batch size 64, 5 broadcasting additions after", "mm_pw", [(64, 10000), (128, 5000), (1024, 250)])
