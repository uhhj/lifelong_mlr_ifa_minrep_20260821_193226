# CCBP-v4 GOAL seed41 D/O/S最终验收

独立验收完成；探索支持条件：{'D-C': False, 'O-C': False, 'S-C': False, 'O-D': True, 'S-O': True}。不自动扩展。

|组别|AUC|FWT|NBT（低好）|OldFinal|FinalAll|
|---|---:|---:|---:|---:|---:|
|C|87.5000|92.5000|7.7778|80.0000|82.5000|
|D|78.3333|86.2500|16.3889|71.6667|72.5000|
|O|87.5000|92.5000|9.4444|76.6667|80.0000|
|S|86.8750|91.2500|5.5556|81.6667|85.0000|

|阶段/任务|C|D|O|S|
|---|---:|---:|---:|---:|
|6/6|90.0|90.0|90.0|90.0|
|7/6|80.0|60.0|70.0|70.0|
|7/7|100.0|100.0|100.0|100.0|
|8/6|50.0|75.0|85.0|45.0|
|8/7|100.0|95.0|100.0|100.0|
|8/8|90.0|80.0|90.0|80.0|
|9/6|40.0|55.0|45.0|45.0|
|9/7|100.0|100.0|100.0|100.0|
|9/8|100.0|60.0|85.0|100.0|
|9/9|90.0|75.0|90.0|95.0|

全部配对SF/FS和负单元见LOCAL_FINAL_AUDIT。离线保持成本因缓存来源不同不能直接跨组当同分布误差；共同专家窗口的误差另列。120训练采集与600正式环境分开计费/计量。C只读复用，非交互等资源。

Single repeatedly developed seed and evaluation initial states; not confirmation or significance.
Existing C had no additional training collection; DOS matched attempt budgets, not equal runtime or retained sample counts.
S filters whole trajectories, not step optimality; empty tasks contribute zero and may lose protection.
Own-policy histories and teachers change across arms; O-S is the full filtering intervention, not isolated label quality.
Offline retention costs use different arm-specific histories; common expert windows are checked separately.
Online cached feature hashes and source-policy replay are verified; no raw-camera/physics replay or causal failure timing is claimed.
