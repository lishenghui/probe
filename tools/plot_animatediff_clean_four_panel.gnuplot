set terminal pngcairo size 3000,760 enhanced font "Sans,15"
set output "artifacts/animatediff_fraq_stack/clean_runtime_benchmark/run_1194031/clean_runtime_four_panel.png"
set multiplot layout 1,4 title "AnimateDiff Runtime LoRA — FRAQ Energy/Rank Trade-offs (GH200, 10 measured runs)" font ",20"
set style fill solid 0.78 border rgb "#333333"
set boxwidth 0.66
set bmargin 4
set grid ytics lc rgb "#dddddd"
set key off
set xtics ("E100" 0, "E95" 1, "E90" 2, "E80" 3, "E70" 4)
set xrange [-0.7:4.7]

set title "Denoising latency"
set ylabel "Seconds / video (lower is better)"
set yrange [21.3:23.6]
set label 1 "23.222" at 0,23.34 center
set label 2 "21.835\n-5.97%" at 1,22.02 center tc rgb "#146c36"
set label 3 "21.698\n-6.56%" at 2,21.89 center tc rgb "#146c36"
set label 4 "21.624\n-6.88%" at 3,21.81 center tc rgb "#146c36"
set label 5 "21.590\n-7.03%" at 4,21.77 center tc rgb "#146c36"
plot '-' using 1:2:5 with boxes lc rgb variable, '-' using 1:2:3:4 with yerrorbars pt 7 ps 0.5 lw 2 lc rgb "#111111"
0 23.22173 23.20730 23.24011 0x666666
1 21.83459 21.82079 21.85238 0x4f8ee8
2 21.69810 21.68398 21.70466 0x4f8ee8
3 21.62358 21.60755 21.63482 0x4f8ee8
4 21.58993 21.58133 21.59443 0x2ca25f
e
0 23.22173 23.20730 23.24011
1 21.83459 21.82079 21.85238
2 21.69810 21.68398 21.70466
3 21.62358 21.60755 21.63482
4 21.58993 21.58133 21.59443
e
unset label

set title "Denoising throughput"
set ylabel "Frames / second (higher is better)"
set yrange [0.68:0.75]
set label 1 "0.689" at 0,0.692 center
set label 2 "0.733\n+6.35%" at 1,0.738 center tc rgb "#146c36"
set label 3 "0.737\n+7.02%" at 2,0.742 center tc rgb "#146c36"
set label 4 "0.740\n+7.39%" at 3,0.745 center tc rgb "#146c36"
set label 5 "0.741\n+7.56%" at 4,0.746 center tc rgb "#146c36"
plot '-' using 1:2:5 with boxes lc rgb variable, '-' using 1:2:3:4 with yerrorbars pt 7 ps 0.5 lw 2 lc rgb "#111111"
0 0.689010 0.688465 0.689438 0x666666
1 0.732782 0.732186 0.733246 0x4f8ee8
2 0.737392 0.737169 0.737872 0x4f8ee8
3 0.739933 0.739548 0.740482 0x4f8ee8
4 0.741086 0.740932 0.741382 0x2ca25f
e
0 0.689010 0.688465 0.689438
1 0.732782 0.732186 0.733246
2 0.737392 0.737169 0.737872
3 0.739933 0.739548 0.740482
4 0.741086 0.740932 0.741382
e
unset label

set title "Mean retained rank"
set ylabel "Rank across 168 modules"
set yrange [0:285]
set label 1 "256.0" at 0,269 center
set label 2 "40.6" at 1,55 center
set label 3 "28.1" at 2,43 center
set label 4 "17.0" at 3,32 center
set label 5 "11.4" at 4,27 center
plot '-' using 1:2:3 with boxes lc rgb variable
0 256.000 0x666666
1 40.565 0x4f8ee8
2 28.137 0x4f8ee8
3 17.048 0x4f8ee8
4 11.387 0x2ca25f
e
unset label

set title "Adapter storage"
set ylabel "Checkpoint size (MB)"
set yrange [0:350]
set label 1 "309.4" at 0,326 center
set label 2 "83.0" at 1,99 center
set label 3 "57.6" at 2,74 center
set label 4 "35.1" at 3,51 center
set label 5 "23.4" at 4,40 center
plot '-' using 1:2:3 with boxes lc rgb variable
0 309.437 0x666666
1 82.981 0x4f8ee8
2 57.622 0x4f8ee8
3 35.127 0x4f8ee8
4 23.444 0x2ca25f
e

unset multiplot
