# 门控图横版排版

按用户要求，将原2×2图改为1×4，顺序为OBJECT散点、OBJECT条件条形、GOAL散点、GOAL条件条形。全部字体使用Times New Roman，接受/拒绝图例与DEBR、对称对数坐标说明均置于整图下方。

从原始门控标量重新绘制，不使用截图重采样数据。OBJECT 27,300与GOAL 20,100个点全部保留，逐块重新核对接受决策；条形比例由相同记录重算。相对差值沿用200*(Q-C)/(abs(Q)+abs(C))，双零记零，对称对数参数不变。无训练、无环境调用。

已目视检查preview.png，无文字裁切、标签重叠。DATA_LAYOUT_AUDIT.json核对所有可见文字的字体及画布边界。原图及原脚本未覆盖。

figures目录提供600dpi PNG、PDF和SVG。PDF/SVG中的散点为600dpi栅格层，其余文字和图形为矢量。

复现：使用安装matplotlib的Python执行make_figure.py，原始证据位置及SHA见SOURCE_MANIFEST.json。
