@echo off
cd /d "%~dp0"

v4.exe ^
  --gens 1000000 ^
  --pop 120 ^
  --n-opponents 8 ^
  --mut 0.03 ^
  --elite 4 ^
  --seed 20260712 ^
  --red-resume returnhome_blocked_seed152_red.npy ^
  --blue-resume returnhome_blocked_seed152_blue.npy ^
  --checkpoint-every 1000 ^
  --benchmark-every 1000 ^
  --benchmark-trials 200 ^
  --max-hours 8 ^
  --out-prefix v4_eventonly_8h_seed20260712 ^
  > v4_eventonly_8h_seed20260712.log ^
  2> v4_eventonly_8h_seed20260712.err.log
