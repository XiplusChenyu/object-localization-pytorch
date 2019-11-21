def iou_calculator(boundary1, boundary2):
    xmax1, ymax1, xmin1, ymin1 = boundary1["xmax"], boundary1["ymax"], boundary1["xmin"], boundary1["ymin"]
    xmax2, ymax2, xmin2, ymin2 = boundary2["xmax"], boundary2["ymax"], boundary2["xmin"], boundary2["ymin"]
    i_xmin = max(xmin1, xmin2)
    i_xmax = min(xmax1, xmax2)
    i_ymin = max(ymin1, ymin2)
    i_ymax = min(ymax1, ymax2)
    if i_xmax <= i_xmin or i_ymax <= i_ymin:
        return 0
    else:
        space1 = (xmax1-xmin1) * (ymax1-ymin1)
        space2 = (xmax2-xmin2) * (ymax2-ymin2)
        i_space = (i_xmax-i_xmin) * (i_ymax-i_ymin)
        return (i_space / ((space1+space2) - i_space)) * 1.0
