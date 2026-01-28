import matplotlib.pyplot as plt


def barplot_annotate_brackets(
    num1,
    num2,
    data,
    center,
    height,
    yerr=None,
    dh=0.05,
    barh=0.05,
    linewidth=1.0,
    fs=13,
    maxasterix=None,
):
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ""
        p = 0.05

        while data < p:
            text += "*"
            p /= 10.0

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = "n. s."

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c="black", linewidth=linewidth)

    kwargs = dict(ha="center", va="bottom")
    if fs is not None:
        kwargs["fontsize"] = fs

    plt.text(*mid, text, **kwargs)


def annotation_text(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."
