set terminal pngcairo size 1800,1200 enhanced font "Sans,15"
set output "artifacts/animatediff_fraq_stack/runtime_latency_benchmark/run_1193395/runtime_rank_latency_comparison.png"
set multiplot layout 2,2 title "AnimateDiff Runtime LoRA: FRAQ Rank vs Performance (GH200, 512x512, 16 frames, 25 steps)" font ",19"
set style fill solid 0.78 border rgb "#333333"
set boxwidth 0.68
set grid ytics lc rgb "#dddddd"
set key off
set xtics ("E100" 0, "E95" 1, "E90" 2, "E80" 3, "E70" 4)
set xrange [-0.7:4.7]

set title "Steady-state latency (lower is better)"
set ylabel "Seconds / video"
set yrange [20:27]
set label 1 "baseline" at 0,25.75 center tc rgb "#444444"
set label 2 "-6.4%" at 1,24.10 center tc rgb "#16703a"
set label 3 "-5.3%" at 2,24.38 center tc rgb "#16703a"
set label 4 "-4.5%" at 3,24.59 center tc rgb "#16703a"
set label 5 "-7.3%" at 4,23.88 center tc rgb "#16703a"
plot '-' using 1:2:3 with boxes lc rgb variable
0 25.2599 0x555555
1 23.6356 0x3b82f6
2 23.9128 0x3b82f6
3 24.1246 0x3b82f6
4 23.4089 0x16a34a
e
unset label

set title "Generation throughput (higher is better)"
set ylabel "Frames / second"
set yrange [0.60:0.71]
set label 1 "+6.9%" at 1,0.683 center tc rgb "#16703a"
set label 2 "+5.6%" at 2,0.675 center tc rgb "#16703a"
set label 3 "+4.7%" at 3,0.669 center tc rgb "#16703a"
set label 4 "+7.9%" at 4,0.690 center tc rgb "#16703a"
plot '-' using 1:2:3 with boxes lc rgb variable
0 0.6334 0x555555
1 0.6769 0x3b82f6
2 0.6691 0x3b82f6
3 0.6632 0x3b82f6
4 0.6835 0x16a34a
e
unset label

set title "Mean retained rank (lower is smaller/faster)"
set ylabel "Mean rank across 168 modules"
set yrange [0:280]
set label 1 "251.0" at 0,263 center
set label 2 "40.6" at 1,53 center
set label 3 "28.1" at 2,41 center
set label 4 "17.0" at 3,30 center
set label 5 "11.4" at 4,24 center
plot '-' using 1:2:3 with boxes lc rgb variable
0 251.012 0x555555
1 40.565 0x3b82f6
2 28.137 0x3b82f6
3 17.048 0x3b82f6
4 11.387 0x16a34a
e
unset label

set title "Adapter checkpoint size"
set ylabel "Size (MB)"
set yrange [0:420]
set label 1 "374.0" at 0,392 center
set label 2 "83.0" at 1,101 center
set label 3 "57.6" at 2,76 center
set label 4 "35.1" at 3,53 center
set label 5 "23.4" at 4,42 center
plot '-' using 1:2:3 with boxes lc rgb variable
0 374.008 0x555555
1 82.981 0x3b82f6
2 57.622 0x3b82f6
3 35.127 0x3b82f6
4 23.444 0x16a34a
e

unset multiplot
